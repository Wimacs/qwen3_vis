/* Qwen3 Chat GUI with Activation Visualization using raylib */

// Avoid Windows API name conflicts with raylib
#if defined(_WIN32)
    #define NOGDI
    #define NOUSER
#endif

#include "raylib.h"
#include "raymath.h"
#include "rlgl.h"

#if defined(_WIN32)
    #undef near
    #undef far
#endif

#include "qwen3.h"
#include <string.h>
#include <stdio.h>
#include <float.h>

#if defined(_WIN32)
    #include <windows.h>
    #include <imm.h>
#endif
// ----------------------------------------------------------------------------
// Configuration

#define SCREEN_WIDTH 1400
#define SCREEN_HEIGHT 800
#define MAX_MESSAGES 100
#define MAX_MESSAGE_LENGTH 4096
#define UI_SCALE 3
#define UI_SCALE_I(x) ((x) * UI_SCALE)
#define BASE_INPUT_BOX_HEIGHT 50
#define BASE_MESSAGE_PADDING 12
#define BASE_FONT_SIZE 20
#define BASE_LINE_HEIGHT 28
#define INPUT_BOX_HEIGHT UI_SCALE_I(BASE_INPUT_BOX_HEIGHT)
#define MESSAGE_PADDING UI_SCALE_I(BASE_MESSAGE_PADDING)
#define FONT_SIZE UI_SCALE_I(BASE_FONT_SIZE)
#define LINE_HEIGHT UI_SCALE_I(BASE_LINE_HEIGHT)

// Layout
#define CHAT_PANEL_WIDTH 450
#define PANEL_MARGIN 10

// Colors
#define BG_COLOR (Color){25, 25, 30, 255}
#define CHAT_BG_COLOR (Color){35, 35, 42, 255}
#define VIS_BG_COLOR (Color){20, 20, 25, 255}
#define INPUT_BG_COLOR (Color){45, 45, 55, 255}
#define USER_MSG_COLOR (Color){50, 90, 140, 255}
#define ASSISTANT_MSG_COLOR (Color){40, 48, 58, 255}
#define TEXT_COLOR (Color){220, 220, 230, 255}
#define PLACEHOLDER_COLOR (Color){100, 100, 110, 255}
#define ACCENT_COLOR (Color){80, 140, 255, 255}
#define TITLE_BG_COLOR (Color){45, 45, 55, 255}

// Global UI font (set in main after loading)
static Font g_ui_font = {0};

// ----------------------------------------------------------------------------
// Message structure

typedef struct {
    char text[MAX_MESSAGE_LENGTH];
    int is_user;
    int is_complete;
} ChatMessage;

typedef struct {
    ChatMessage messages[MAX_MESSAGES];
    int message_count;
    char input_buffer[MAX_MESSAGE_LENGTH];
    int input_length;
    float scroll_offset;
    float target_scroll;
    int is_generating;
} ChatUI;

// ----------------------------------------------------------------------------
// Activation Visualization

typedef enum {
    VIS_ATTENTION,      // Attention scores (n_heads x seq_len)
    VIS_HIDDEN_STATE,   // Hidden state x (dim)
    VIS_QUERY,          // Query vectors (n_heads x head_dim)
    VIS_LOGITS,         // Output logits (vocab_size) - top tokens
    VIS_FFN_ACTIVATION, // FFN hidden (hidden_dim)
    VIS_NETWORK_FULL,   // Full network visualization (bbycroft style)
    VIS_COUNT
} VisType;

typedef struct {
    Texture2D texture;
    Image image;
    int width;
    int height;
    float scale;
    Vector2 offset;
    float min_val;
    float max_val;
    int valid;
} ActivationVis;

// ----------------------------------------------------------------------------
// Full Network Visualization (bbycroft style)

// Global quantization group size (from qwen3.c)
extern int GS;

// Node types for the network graph
typedef enum {
    NODE_TYPE_ACTIVATION,   // Activation vector (float*)
    NODE_TYPE_WEIGHT,       // Weight matrix (QuantizedTensor)
    NODE_TYPE_OPERATION     // Operation node (RMSNorm, SiLU, etc.)
} NodeType;

// Node IDs for the network graph
typedef enum {
    VNODE_X,            // Input hidden state
    VNODE_RMSNORM1,     // First RMSNorm (attention)
    VNODE_XB,           // Normalized hidden state
    VNODE_WQ,           // Query weight matrix
    VNODE_WK,           // Key weight matrix
    VNODE_WV,           // Value weight matrix
    VNODE_Q,            // Query vectors
    VNODE_K,            // Key vectors
    VNODE_V,            // Value vectors
    VNODE_ATT,          // Attention scores
    VNODE_ATTV,         // Attention * V output
    VNODE_WO,           // Output projection weight
    VNODE_ATTN_OUT,     // Attention output
    VNODE_RES1,         // First residual add
    VNODE_RMSNORM2,     // Second RMSNorm (FFN)
    VNODE_XB2,          // Normalized for FFN
    VNODE_W1,           // FFN up-projection weight
    VNODE_W3,           // FFN gate-projection weight
    VNODE_HB,           // FFN hidden (up)
    VNODE_HB2,          // FFN hidden (gate)
    VNODE_SILU,         // SiLU gate operation
    VNODE_W2,           // FFN down-projection weight
    VNODE_FFN_OUT,      // FFN output
    VNODE_RES2,         // Second residual add
    VNODE_XOUT,         // Layer output
    VNODE_COUNT
} VisNodeId;

// Neuron grid - each neuron is one colored square
typedef struct {
    float *data;            // Pointer to activation data (float)
    QuantizedTensor *qdata; // Pointer to quantized weight data
    int rows;               // Grid rows
    int cols;               // Grid cols
    int is_quantized;       // Whether data is quantized
    float min_val;          // Cached min value for coloring
    float max_val;          // Cached max value for coloring
    int values_computed;    // Whether min/max have been computed
} NeuronGrid;

// Visualization node
typedef struct {
    const char *name;       // Short name (e.g., "x", "Wq")
    const char *label;      // Display label
    NodeType type;          // Type of node
    NeuronGrid grid;        // Data grid
    Vector2 pos;            // Position in layout
    float width;            // Display width
    float height;           // Display height
    int visible;            // Whether to render
    int collapsed;          // Whether weight matrix is collapsed
    int highlight;          // Highlight level (for animation)
} VisNode;

// Data flow edge (with animation)
typedef struct {
    VisNodeId from_node;
    VisNodeId to_node;
    Color color;
    float animation_t;      // 0-1 animation progress
    int active;             // Whether data is flowing
} DataFlowEdge;

// Layer view state
typedef struct {
    int current_layer;      // Current layer being viewed (0 to n_layers-1)
    float zoom;             // Zoom level
    Vector2 pan;            // Pan offset
    int show_weights;       // Whether to show weight matrices
    int show_activations;   // Whether to show activations
    float animation_speed;  // Animation speed multiplier
    int animation_step;     // Current step in forward pass animation
    float step_timer;       // Timer for auto-advancing animation
} LayerView;

// ----------------------------------------------------------------------------
// 3D Visualization Structures

// LOD levels for performance optimization
typedef enum {
    LOD_FULL,       // All voxels (< 10 units from camera)
    LOD_SUBSAMPLE,  // Every 2nd voxel (10-30 units)
    LOD_ICON        // Single representative cube (> 30 units)
} LODLevel;

// 3D node transform
typedef struct {
    Vector3 position;      // 3D world coordinates
    Vector3 scale;         // Node bounding box dimensions
    float z_layer;         // Z-axis layer index
    BoundingBox bbox;      // For ray picking
} VisNode3DTransform;

// Free-flight camera controller
typedef struct {
    Camera3D camera;       // Raylib camera structure
    Vector3 velocity;      // Movement velocity
    float yaw;             // Horizontal rotation (radians)
    float pitch;           // Vertical rotation (radians)
    float move_speed;      // Movement speed (default 10.0 units/sec)
    float mouse_sensitivity; // Mouse sensitivity (0.003 radians/pixel)
    int locked;            // Mouse captured for camera control
} FreeFlightCamera;

// Voxel instance (single point/cube representing one activation value)
typedef struct {
    Vector3 position;      // Voxel world position
    Color color;           // Color (Inferno colormap)
    float value;           // Original activation value
    float size;            // Display size (for point rendering)
} VoxelInstance;

// Voxel cache (with LOD support)
typedef struct {
    VoxelInstance* instances;
    int instance_count;
    int capacity;
    int dirty;             // Needs rebuild
} VoxelCache;

// Texture-based rendering for full-resolution weights
typedef struct {
    Texture2D texture;
    Image image;
    int valid;
    int dirty;
} NodeTextureCache;

// Full network visualization state
typedef struct {
    // === 2D Visualization Fields ===
    VisNode nodes[VNODE_COUNT];
    int node_count;
    DataFlowEdge edges[32];
    int edge_count;
    LayerView view;
    int initialized;
    // Hover state (2D)
    VisNodeId hover_node;
    int hover_row;
    int hover_col;
    float hover_value;
    int has_hover;

    // === 3D Visualization Fields ===
    VisNode3DTransform transforms3d[VNODE_COUNT];
    VoxelCache voxel_caches[VNODE_COUNT];
    NodeTextureCache texture_caches[VNODE_COUNT];  // Texture-based rendering
    FreeFlightCamera camera;
    int use_3d_mode;       // Toggle between 2D and 3D rendering
    int use_textures;      // Use texture rendering instead of voxels

    // Ray picking state (3D)
    Ray mouse_ray;
    VisNodeId picked_node;
    float picked_value;
    int has_pick;
} NetworkVis;

// ----------------------------------------------------------------------------
// Forward declarations for 3D functions
static void InitFreeFlightCamera(FreeFlightCamera* cam, Vector3 start_pos, Vector3 look_at);
static void UpdateFreeFlightCamera(FreeFlightCamera* cam, float dt);
static Font GetUIFont(void);
static int MeasureTextUI(const char *text, int fontSize);
static void DrawTextUI(const char *text, float x, float y, int fontSize, Color color);

// ----------------------------------------------------------------------------
// Color mapping functions

// Inferno colormap (good for heatmaps)
Color FloatToColorInferno(float t) {
    // t should be 0 to 1
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    
    // Approximate inferno colormap
    float r = t < 0.5f ? (2.0f * t * 0.9f) : (0.9f + 0.1f * (2.0f * t - 1.0f));
    float g = t < 0.3f ? 0 : (t < 0.7f ? (t - 0.3f) * 2.0f : 0.8f + 0.2f * (t - 0.7f) / 0.3f);
    float b = t < 0.5f ? (0.3f + t * 0.6f) : (0.6f - (t - 0.5f) * 1.2f);
    
    if (r < 0) r = 0; if (r > 1) r = 1;
    if (g < 0) g = 0; if (g > 1) g = 1;
    if (b < 0) b = 0; if (b > 1) b = 1;
    
    return (Color){(unsigned char)(r * 255), (unsigned char)(g * 255), (unsigned char)(b * 255), 255};
}

// Blue-white-red diverging colormap
Color FloatToColorDiverging(float val, float minVal, float maxVal) {
    float range = maxVal - minVal;
    if (range < 1e-6f) range = 1.0f;
    float t = (val - minVal) / range;  // 0 to 1
    
    float mid = (0 - minVal) / range;  // where zero is
    if (mid < 0) mid = 0;
    if (mid > 1) mid = 1;
    
    if (t < mid) {
        // Negative: blue to white
        float s = t / mid;
        return (Color){
            (unsigned char)(s * 255),
            (unsigned char)(s * 255),
            255,
            255
        };
    } else {
        // Positive: white to red
        float s = (t - mid) / (1.0f - mid);
        return (Color){
            255,
            (unsigned char)((1 - s) * 255),
            (unsigned char)((1 - s) * 255),
            255
        };
    }
}

// Create/update attention visualization
void UpdateAttentionVis(ActivationVis *vis, RunState *state, Config *cfg, int current_pos) {
    int n_heads = cfg->n_heads;
    int seq_len = current_pos + 1;
    
    if (seq_len < 1) seq_len = 1;
    if (seq_len > cfg->seq_len) seq_len = cfg->seq_len;
    
    // Resize if needed
    if (vis->width != seq_len || vis->height != n_heads || !vis->valid) {
        if (vis->valid) {
            UnloadTexture(vis->texture);
            UnloadImage(vis->image);
        }
        vis->width = seq_len;
        vis->height = n_heads;
        vis->image = GenImageColor(seq_len, n_heads, BLACK);
        vis->texture = LoadTextureFromImage(vis->image);
        SetTextureFilter(vis->texture, TEXTURE_FILTER_POINT);
        vis->valid = 1;
    }
    
    // Find min/max for normalization
    vis->min_val = FLT_MAX;
    vis->max_val = -FLT_MAX;
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < seq_len; t++) {
            float val = state->att[h * cfg->seq_len + t];
            if (val < vis->min_val) vis->min_val = val;
            if (val > vis->max_val) vis->max_val = val;
        }
    }
    
    // Fill image
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < seq_len; t++) {
            float val = state->att[h * cfg->seq_len + t];
            float normalized = (val - vis->min_val) / (vis->max_val - vis->min_val + 1e-6f);
            Color c = FloatToColorInferno(normalized);
            ImageDrawPixel(&vis->image, t, h, c);
        }
    }
    
    UpdateTexture(vis->texture, vis->image.data);
}

// Create/update hidden state visualization (reshape to 2D)
void UpdateHiddenStateVis(ActivationVis *vis, float *data, int size) {
    // Reshape 1D vector to approximate square
    int side = (int)sqrtf((float)size);
    while (size % side != 0 && side > 1) side--;
    int width = size / side;
    int height = side;
    
    if (vis->width != width || vis->height != height || !vis->valid) {
        if (vis->valid) {
            UnloadTexture(vis->texture);
            UnloadImage(vis->image);
        }
        vis->width = width;
        vis->height = height;
        vis->image = GenImageColor(width, height, BLACK);
        vis->texture = LoadTextureFromImage(vis->image);
        SetTextureFilter(vis->texture, TEXTURE_FILTER_POINT);
        vis->valid = 1;
    }
    
    // Find min/max
    vis->min_val = FLT_MAX;
    vis->max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (data[i] < vis->min_val) vis->min_val = data[i];
        if (data[i] > vis->max_val) vis->max_val = data[i];
    }
    
    // Fill image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            Color c = FloatToColorDiverging(data[idx], vis->min_val, vis->max_val);
            ImageDrawPixel(&vis->image, x, y, c);
        }
    }
    
    UpdateTexture(vis->texture, vis->image.data);
}

// Create/update query visualization (n_heads x head_dim)
void UpdateQueryVis(ActivationVis *vis, float *q, int n_heads, int head_dim) {
    int width = head_dim;
    int height = n_heads;
    
    if (vis->width != width || vis->height != height || !vis->valid) {
        if (vis->valid) {
            UnloadTexture(vis->texture);
            UnloadImage(vis->image);
        }
        vis->width = width;
        vis->height = height;
        vis->image = GenImageColor(width, height, BLACK);
        vis->texture = LoadTextureFromImage(vis->image);
        SetTextureFilter(vis->texture, TEXTURE_FILTER_POINT);
        vis->valid = 1;
    }
    
    // Find min/max
    int total = n_heads * head_dim;
    vis->min_val = FLT_MAX;
    vis->max_val = -FLT_MAX;
    for (int i = 0; i < total; i++) {
        if (q[i] < vis->min_val) vis->min_val = q[i];
        if (q[i] > vis->max_val) vis->max_val = q[i];
    }
    
    // Fill image
    for (int h = 0; h < n_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
            int idx = h * head_dim + d;
            Color c = FloatToColorDiverging(q[idx], vis->min_val, vis->max_val);
            ImageDrawPixel(&vis->image, d, h, c);
        }
    }
    
    UpdateTexture(vis->texture, vis->image.data);
}

// Create/update logits visualization (show as bar chart in texture)
void UpdateLogitsVis(ActivationVis *vis, float *logits, int vocab_size, int top_k) {
    int width = top_k;
    int height = 64;  // Bar height
    
    if (vis->width != width || vis->height != height || !vis->valid) {
        if (vis->valid) {
            UnloadTexture(vis->texture);
            UnloadImage(vis->image);
        }
        vis->width = width;
        vis->height = height;
        vis->image = GenImageColor(width, height, (Color){30, 30, 35, 255});
        vis->texture = LoadTextureFromImage(vis->image);
        SetTextureFilter(vis->texture, TEXTURE_FILTER_POINT);
        vis->valid = 1;
    }
    
    // Find top-k logits
    int *top_indices = malloc(top_k * sizeof(int));
    float *top_values = malloc(top_k * sizeof(float));
    for (int i = 0; i < top_k; i++) {
        top_values[i] = -FLT_MAX;
        top_indices[i] = 0;
    }
    
    for (int i = 0; i < vocab_size; i++) {
        float val = logits[i];
        // Insert into sorted list if bigger than smallest
        if (val > top_values[top_k - 1]) {
            int j = top_k - 1;
            while (j > 0 && val > top_values[j - 1]) {
                top_values[j] = top_values[j - 1];
                top_indices[j] = top_indices[j - 1];
                j--;
            }
            top_values[j] = val;
            top_indices[j] = i;
        }
    }
    
    vis->min_val = top_values[top_k - 1];
    vis->max_val = top_values[0];
    
    // Clear and draw bars
    ImageClearBackground(&vis->image, (Color){30, 30, 35, 255});
    for (int i = 0; i < top_k; i++) {
        float normalized = (top_values[i] - vis->min_val) / (vis->max_val - vis->min_val + 1e-6f);
        int barHeight = (int)(normalized * (height - 2));
        Color c = FloatToColorInferno(normalized);
        for (int y = height - 1 - barHeight; y < height; y++) {
            ImageDrawPixel(&vis->image, i, y, c);
        }
    }
    
    UpdateTexture(vis->texture, vis->image.data);
    
    free(top_indices);
    free(top_values);
}

void FreeActivationVis(ActivationVis *vis) {
    if (vis->valid) {
        UnloadTexture(vis->texture);
        UnloadImage(vis->image);
        vis->valid = 0;
    }
}

// ----------------------------------------------------------------------------
// 3D Voxel Generation and Rendering - OPTIMIZED VERSION

// Forward declarations for functions defined later
static float GetGridValue(NeuronGrid *grid, int row, int col);
static void UpdateNetworkVisData(NetworkVis *nv, RunState *state, TransformerWeights *weights, 
                                  Config *cfg, int current_pos);

// Performance constants
#define MAX_VOXELS_PER_NODE 2048      // Hard limit per node
#define POINT_RENDER_THRESHOLD 500    // Use points instead of cubes above this count
#define FRUSTUM_PADDING 2.0f          // Padding for frustum culling

// Simple frustum culling check - DISABLED for now to ensure visibility
static int IsInFrustum(Vector3 center, float radius, Camera3D cam, float screenW, float screenH) {
    // Temporarily disabled frustum culling to debug visibility
    // Just do simple distance check
    float distSq = Vector3DistanceSqr(cam.position, center);
    
    // Only cull if extremely far
    if (distSq > 20000.0f) return 0;
    
    return 1;
}

// Get LOD parameters based on camera distance
static void GetLODParams(float cameraDist, int* outStep, int* outMaxVoxels, float* outPointSize) {
    if (cameraDist < 15.0f) {
        *outStep = 1;
        *outMaxVoxels = 2048;
        *outPointSize = 3.0f;
    } else if (cameraDist < 30.0f) {
        *outStep = 2;
        *outMaxVoxels = 1024;
        *outPointSize = 4.0f;
    } else if (cameraDist < 50.0f) {
        *outStep = 4;
        *outMaxVoxels = 512;
        *outPointSize = 5.0f;
    } else {
        *outStep = 8;
        *outMaxVoxels = 256;
        *outPointSize = 6.0f;
    }
}

// Build voxel cache for a neuron grid with aggressive LOD
static void BuildVoxelCache(VoxelCache* cache, NeuronGrid* grid, Vector3 basePos, 
                             float voxelSize, float cameraDist) {
    if (!cache->dirty && cache->instances != NULL) return;
    
    // Free old cache
    if (cache->instances != NULL) {
        free(cache->instances);
        cache->instances = NULL;
    }
    
    int rows = grid->rows;
    int cols = grid->cols;
    if (rows <= 0 || cols <= 0) return;
    
    // Determine LOD parameters
    int step;
    int maxVoxels;
    float pointSize;
    GetLODParams(cameraDist, &step, &maxVoxels, &pointSize);
    
    // Calculate grid dimensions after subsampling
    int gridW = (cols + step - 1) / step;
    int gridH = (rows + step - 1) / step;
    int totalCells = gridW * gridH;
    
    // Further reduce if still too many
    while (totalCells > maxVoxels && step < 16) {
        step *= 2;
        gridW = (cols + step - 1) / step;
        gridH = (rows + step - 1) / step;
        totalCells = gridW * gridH;
    }
    
    // Allocate cache
    int allocCount = totalCells > maxVoxels ? maxVoxels : totalCells;
    cache->instances = (VoxelInstance*)malloc(allocCount * sizeof(VoxelInstance));
    cache->capacity = allocCount;
    cache->instance_count = 0;
    
    if (!cache->instances) return;
    
    float range = grid->max_val - grid->min_val;
    if (range < 1e-6f) range = 1.0f;
    
    // Center the grid
    float totalWidth = cols * voxelSize;
    float totalHeight = rows * voxelSize;
    float offsetX = -totalWidth / 2.0f;
    float offsetY = -totalHeight / 2.0f;
    
    // Generate voxels with skip pattern for very large grids
    int skipInterval = totalCells > maxVoxels ? (totalCells + maxVoxels - 1) / maxVoxels : 1;
    int counter = 0;
    
    for (int r = 0; r < rows && cache->instance_count < allocCount; r += step) {
        for (int c = 0; c < cols && cache->instance_count < allocCount; c += step) {
            counter++;
            if (skipInterval > 1 && (counter % skipInterval) != 0) continue;
            
            float val = GetGridValue(grid, r, c);
            float t = (val - grid->min_val) / range;
            
            VoxelInstance* v = &cache->instances[cache->instance_count];
            v->position = (Vector3){
                basePos.x + offsetX + c * voxelSize,
                basePos.y + offsetY + r * voxelSize,
                basePos.z
            };
            v->color = FloatToColorInferno(t);
            v->value = val;
            v->size = pointSize;  // Store point size in value field
            
            cache->instance_count++;
        }
    }
    
    cache->dirty = 0;
}

// Free voxel cache
static void FreeVoxelCache(VoxelCache* cache) {
    if (cache->instances) {
        free(cache->instances);
        cache->instances = NULL;
    }
    cache->instance_count = 0;
    cache->capacity = 0;
    cache->dirty = 1;
}

// Create/update texture from neuron grid (for high-performance rendering)
static void UpdateNodeTexture(NodeTextureCache* tc, NeuronGrid* grid) {
    if (!tc->dirty && tc->valid) return;
    
    // Free old texture
    if (tc->valid) {
        UnloadTexture(tc->texture);
        UnloadImage(tc->image);
    }
    
    int rows = grid->rows;
    int cols = grid->cols;
    if (rows <= 0 || cols <= 0) {
        tc->valid = 0;
        return;
    }
    
    // Limit texture size to avoid GPU memory issues
    int maxTexSize = 1024;
    int texW = cols;
    int texH = rows;
    
    if (texW > maxTexSize || texH > maxTexSize) {
        // Subsample to fit
        float ratioW = (float)maxTexSize / texW;
        float ratioH = (float)maxTexSize / texH;
        float ratio = fminf(ratioW, ratioH);
        texW = (int)(texW * ratio);
        texH = (int)(texH * ratio);
    }
    
    // Ensure minimum size
    if (texW < 1) texW = 1;
    if (texH < 1) texH = 1;
    
    // Create image
    tc->image = GenImageColor(texW, texH, BLACK);
    
    // Fill with color-mapped data
    float range = grid->max_val - grid->min_val;
    if (range < 1e-6f) range = 1.0f;
    
    int rowStep = rows / texH;
    int colStep = cols / texW;
    if (rowStep < 1) rowStep = 1;
    if (colStep < 1) colStep = 1;
    
    for (int y = 0; y < texH; y++) {
        for (int x = 0; x < texW; x++) {
            int r = y * rowStep;
            int c = x * colStep;
            if (r >= rows) r = rows - 1;
            if (c >= cols) c = cols - 1;
            
            float val = GetGridValue(grid, r, c);
            float t = (val - grid->min_val) / range;
            Color color = FloatToColorInferno(t);
            
            ImageDrawPixel(&tc->image, x, y, color);
        }
    }
    
    tc->texture = LoadTextureFromImage(tc->image);
    tc->valid = 1;
    tc->dirty = 0;
}

// Free node texture cache
static void FreeNodeTextureCache(NodeTextureCache* tc) {
    if (tc->valid) {
        UnloadTexture(tc->texture);
        UnloadImage(tc->image);
        tc->valid = 0;
    }
    tc->dirty = 1;
}

// Calculate 3D layout positions for all nodes
static void LayoutNetworkVis3D(NetworkVis* nv) {
    // Define Z-layers for different stages of computation
    const float zInput = 0.0f;
    const float zNorm1 = 4.0f;
    const float zWeightsQKV = 8.0f;
    const float zQKV = 12.0f;
    const float zAttention = 16.0f;
    const float zAttnOut = 20.0f;
    const float zRes1 = 24.0f;
    const float zNorm2 = 28.0f;
    const float zFFN1 = 32.0f;
    const float zFFN2 = 36.0f;
    const float zFFNOut = 40.0f;
    const float zRes2 = 44.0f;
    const float zOutput = 48.0f;
    
    float xLeft = -8.0f;
    float xCenter = 0.0f;
    float xRight = 8.0f;
    float xFarLeft = -12.0f;
    float xFarRight = 12.0f;
    
    // Input
    nv->transforms3d[VNODE_X].position = (Vector3){xCenter, 0, zInput};
    nv->transforms3d[VNODE_X].scale = (Vector3){4, 4, 1};
    
    // RMSNorm 1
    nv->transforms3d[VNODE_RMSNORM1].position = (Vector3){xCenter, 0, zNorm1};
    nv->transforms3d[VNODE_RMSNORM1].scale = (Vector3){2, 2, 1};
    
    nv->transforms3d[VNODE_XB].position = (Vector3){xCenter, 0, zNorm1 + 2};
    nv->transforms3d[VNODE_XB].scale = (Vector3){4, 4, 1};
    
    // QKV weights and vectors
    nv->transforms3d[VNODE_WQ].position = (Vector3){xFarLeft, 4, zWeightsQKV};
    nv->transforms3d[VNODE_WQ].scale = (Vector3){2, 4, 1};
    
    nv->transforms3d[VNODE_WK].position = (Vector3){xLeft, 4, zWeightsQKV};
    nv->transforms3d[VNODE_WK].scale = (Vector3){2, 4, 1};
    
    nv->transforms3d[VNODE_WV].position = (Vector3){xCenter, 4, zWeightsQKV};
    nv->transforms3d[VNODE_WV].scale = (Vector3){2, 4, 1};
    
    nv->transforms3d[VNODE_Q].position = (Vector3){xFarLeft, 4, zQKV};
    nv->transforms3d[VNODE_Q].scale = (Vector3){4, 2, 1};
    
    nv->transforms3d[VNODE_K].position = (Vector3){xLeft, 4, zQKV};
    nv->transforms3d[VNODE_K].scale = (Vector3){3, 2, 1};
    
    nv->transforms3d[VNODE_V].position = (Vector3){xCenter, 4, zQKV};
    nv->transforms3d[VNODE_V].scale = (Vector3){3, 2, 1};
    
    // Attention
    nv->transforms3d[VNODE_ATT].position = (Vector3){xCenter, -2, zAttention};
    nv->transforms3d[VNODE_ATT].scale = (Vector3){6, 4, 1};
    
    nv->transforms3d[VNODE_ATTV].position = (Vector3){xCenter, 4, zAttnOut - 2};
    nv->transforms3d[VNODE_ATTV].scale = (Vector3){4, 2, 1};
    
    // Output projection
    nv->transforms3d[VNODE_WO].position = (Vector3){xRight, 4, zAttnOut - 1};
    nv->transforms3d[VNODE_WO].scale = (Vector3){2, 4, 1};
    
    nv->transforms3d[VNODE_ATTN_OUT].position = (Vector3){xCenter, 0, zAttnOut};
    nv->transforms3d[VNODE_ATTN_OUT].scale = (Vector3){4, 4, 1};
    
    // Residual 1
    nv->transforms3d[VNODE_RES1].position = (Vector3){xCenter, 0, zRes1};
    nv->transforms3d[VNODE_RES1].scale = (Vector3){3, 3, 1};
    
    // FFN path
    nv->transforms3d[VNODE_RMSNORM2].position = (Vector3){xCenter, 0, zNorm2};
    nv->transforms3d[VNODE_RMSNORM2].scale = (Vector3){2, 2, 1};
    
    nv->transforms3d[VNODE_XB2].position = (Vector3){xCenter, 0, zNorm2 + 2};
    nv->transforms3d[VNODE_XB2].scale = (Vector3){4, 4, 1};
    
    // FFN weights
    nv->transforms3d[VNODE_W1].position = (Vector3){xLeft, -4, zFFN1};
    nv->transforms3d[VNODE_W1].scale = (Vector3){2, 4, 1};
    
    nv->transforms3d[VNODE_W3].position = (Vector3){xRight, -4, zFFN1};
    nv->transforms3d[VNODE_W3].scale = (Vector3){2, 4, 1};
    
    nv->transforms3d[VNODE_HB].position = (Vector3){xLeft, -4, zFFN1 + 2};
    nv->transforms3d[VNODE_HB].scale = (Vector3){4, 3, 1};
    
    nv->transforms3d[VNODE_HB2].position = (Vector3){xRight, -4, zFFN1 + 2};
    nv->transforms3d[VNODE_HB2].scale = (Vector3){4, 3, 1};
    
    // SiLU gate
    nv->transforms3d[VNODE_SILU].position = (Vector3){xCenter, -4, zFFN2};
    nv->transforms3d[VNODE_SILU].scale = (Vector3){3, 3, 1};
    
    nv->transforms3d[VNODE_W2].position = (Vector3){xCenter, -4, zFFN2 + 1.5f};
    nv->transforms3d[VNODE_W2].scale = (Vector3){2, 4, 1};
    
    nv->transforms3d[VNODE_FFN_OUT].position = (Vector3){xCenter, 0, zFFNOut};
    nv->transforms3d[VNODE_FFN_OUT].scale = (Vector3){4, 4, 1};
    
    // Residual 2 and output
    nv->transforms3d[VNODE_RES2].position = (Vector3){xCenter, 0, zRes2};
    nv->transforms3d[VNODE_RES2].scale = (Vector3){3, 3, 1};
    
    nv->transforms3d[VNODE_XOUT].position = (Vector3){xCenter, 0, zOutput};
    nv->transforms3d[VNODE_XOUT].scale = (Vector3){4, 4, 1};
    
    // Update bounding boxes
    for (int i = 0; i < VNODE_COUNT; i++) {
        Vector3 pos = nv->transforms3d[i].position;
        Vector3 scale = nv->transforms3d[i].scale;
        nv->transforms3d[i].bbox = (BoundingBox){
            (Vector3){pos.x - scale.x/2, pos.y - scale.y/2, pos.z - 0.5f},
            (Vector3){pos.x + scale.x/2, pos.y + scale.y/2, pos.z + 0.5f}
        };
    }
}

// Draw voxels efficiently - use small cubes which are more reliable than billboards
static void DrawVoxelsAsPoints(VoxelInstance* voxels, int count, Camera3D cam) {
    if (count <= 0 || !voxels) return;
    
    // Always use small cubes - more reliable across different GPUs
    // Batch by proximity to reduce draw call overhead
    for (int i = 0; i < count; i++) {
        VoxelInstance* v = &voxels[i];
        // Scale based on LOD level (stored in v->size)
        float s = v->size * 0.04f;
        DrawCube(v->position, s, s, s * 0.5f, v->color);
    }
}

// Draw 3D edge/connection between nodes
static void Draw3DEdge(Vector3 from, Vector3 to, Color color, float animationT, int active) {
    // Draw line connection
    float thickness = active ? 0.15f : 0.08f;
    DrawLine3D(from, to, color);
    
    // Draw animated particle if active
    if (active && animationT >= 0 && animationT <= 1) {
        Vector3 particlePos = {
            from.x + (to.x - from.x) * animationT,
            from.y + (to.y - from.y) * animationT,
            from.z + (to.z - from.z) * animationT
        };
        DrawSphere(particlePos, 0.2f, WHITE);
    }
}

// Ray picking for 3D voxels
static void UpdateRayPicking3D(NetworkVis* nv, Camera3D camera) {
    Ray ray = GetScreenToWorldRay(GetMousePosition(), camera);
    nv->mouse_ray = ray;
    
    nv->has_pick = 0;
    nv->picked_node = -1;
    
    float closestDist = 1000.0f;
    
    for (int i = 0; i < VNODE_COUNT; i++) {
        if (!nv->nodes[i].visible) continue;
        
        // Check intersection with node bounding box
        RayCollision collision = GetRayCollisionBox(ray, nv->transforms3d[i].bbox);
        
        if (collision.hit && collision.distance < closestDist) {
            closestDist = collision.distance;
            nv->picked_node = i;
            nv->has_pick = 1;
            
            // Calculate approximate grid position from hit point
            Vector3 hitPoint = collision.point;
            Vector3 pos = nv->transforms3d[i].position;
            Vector3 scale = nv->transforms3d[i].scale;
            
            // Simple mapping to grid coordinates
            NeuronGrid* grid = &nv->nodes[i].grid;
            if (grid->rows > 0 && grid->cols > 0) {
                float nx = (hitPoint.x - (pos.x - scale.x/2)) / scale.x;
                float ny = (hitPoint.y - (pos.y - scale.y/2)) / scale.y;
                nv->hover_row = (int)(ny * grid->rows);
                nv->hover_col = (int)(nx * grid->cols);
                if (nv->hover_row < 0) nv->hover_row = 0;
                if (nv->hover_row >= grid->rows) nv->hover_row = grid->rows - 1;
                if (nv->hover_col < 0) nv->hover_col = 0;
                if (nv->hover_col >= grid->cols) nv->hover_col = grid->cols - 1;
                nv->hover_value = GetGridValue(grid, nv->hover_row, nv->hover_col);
            }
        }
    }
}

// Render 3D network using VOXEL mode (fallback)
static void DrawNetworkVis3D_Voxel(NetworkVis* nv, Config* cfg, Camera3D cam, int showWeights, int* nodeVisible) {
    // Batch render: collect all visible voxels first
    static VoxelInstance* visibleVoxels = NULL;
    static int visibleCapacity = 0;
    int visibleCount = 0;
    
    // Ensure buffer capacity
    int maxPossible = VNODE_COUNT * MAX_VOXELS_PER_NODE;
    if (visibleCapacity < maxPossible) {
        visibleVoxels = (VoxelInstance*)realloc(visibleVoxels, maxPossible * sizeof(VoxelInstance));
        visibleCapacity = maxPossible;
    }
    
    float cameraDist = Vector3Distance(cam.position, (Vector3){0, 0, 24});
    
    // First pass: update caches and cull invisible nodes
    for (int i = 0; i < VNODE_COUNT; i++) {
        if (!nv->nodes[i].visible) continue;
        if (nv->nodes[i].type == NODE_TYPE_WEIGHT && !showWeights) continue;
        
        VisNode* node = &nv->nodes[i];
        VisNode3DTransform* trans = &nv->transforms3d[i];
        VoxelCache* cache = &nv->voxel_caches[i];
        
        // Frustum culling for operations
        if (node->type == NODE_TYPE_OPERATION) {
            if (IsInFrustum(trans->position, 0.5f, cam, GetScreenWidth(), GetScreenHeight())) {
                nodeVisible[i] = 1;
                DrawSphere(trans->position, 0.5f, (Color){150, 150, 170, 255});
            }
            continue;
        }
        
        // Skip if no grid data
        if (node->grid.rows <= 0 || node->grid.cols <= 0) continue;
        
        // Frustum culling for voxel nodes
        float nodeRadius = fmaxf(trans->scale.x, trans->scale.y) * 0.7f;
        if (!IsInFrustum(trans->position, nodeRadius, cam, GetScreenWidth(), GetScreenHeight())) {
            continue;
        }
        nodeVisible[i] = 1;
        
        // Rebuild cache if dirty
        if (cache->dirty || cache->instances == NULL) {
            BuildVoxelCache(cache, &node->grid, trans->position, 0.15f, cameraDist);
        }
        
        // If no voxels, draw placeholder
        if (!cache->instances || cache->instance_count == 0) {
            DrawCube(trans->position, 1.0f, 1.0f, 1.0f, (Color){100, 100, 100, 128});
            DrawCubeWires(trans->position, 1.0f, 1.0f, 1.0f, WHITE);
            continue;
        }
        
        // Highlight picked node
        int isPicked = (nv->has_pick && nv->picked_node == i);
        
        // Add voxels to batch
        int toAdd = cache->instance_count;
        if (visibleCount + toAdd > visibleCapacity) {
            toAdd = visibleCapacity - visibleCount;
        }
        
        for (int v = 0; v < toAdd; v++) {
            VoxelInstance* src = &cache->instances[v];
            VoxelInstance* dst = &visibleVoxels[visibleCount++];
            *dst = *src;
            
            if (isPicked) {
                dst->color.r = (unsigned char)fminf(255, dst->color.r * 1.3f);
                dst->color.g = (unsigned char)fminf(255, dst->color.g * 1.3f);
                dst->color.b = (unsigned char)fminf(255, dst->color.b * 1.3f);
            }
        }
        
        // Draw bounding box for weight nodes
        if (node->type == NODE_TYPE_WEIGHT || isPicked) {
            DrawBoundingBox(trans->bbox, isPicked ? YELLOW : (Color){100, 100, 120, 100});
        }
    }
    
    // Batch draw all voxels
    DrawVoxelsAsPoints(visibleVoxels, visibleCount, cam);
}

// Draw a textured quad fixed in 3D space (facing +Z direction, not camera)
static void DrawTexturedQuadFixed(Texture2D texture, Vector3 position, float width, float height, Color tint) {
    // Calculate corners of a quad facing +Z (upright in XY plane)
    float w = width * 0.5f;
    float h = height * 0.5f;
    
    Vector3 topLeft = {position.x - w, position.y + h, position.z};
    Vector3 topRight = {position.x + w, position.y + h, position.z};
    Vector3 bottomRight = {position.x + w, position.y - h, position.z};
    Vector3 bottomLeft = {position.x - w, position.y - h, position.z};
    
    // Use raylib's rlgl to draw textured quad
    rlSetTexture(texture.id);
    
    rlBegin(RL_QUADS);
        rlColor4ub(tint.r, tint.g, tint.b, tint.a);
        
        // Normal facing +Z
        rlNormal3f(0.0f, 0.0f, 1.0f);
        
        // Top-left
        rlTexCoord2f(0.0f, 0.0f);
        rlVertex3f(topLeft.x, topLeft.y, topLeft.z);
        
        // Bottom-left
        rlTexCoord2f(0.0f, 1.0f);
        rlVertex3f(bottomLeft.x, bottomLeft.y, bottomLeft.z);
        
        // Bottom-right
        rlTexCoord2f(1.0f, 1.0f);
        rlVertex3f(bottomRight.x, bottomRight.y, bottomRight.z);
        
        // Top-right
        rlTexCoord2f(1.0f, 0.0f);
        rlVertex3f(topRight.x, topRight.y, topRight.z);
    rlEnd();
    
    rlSetTexture(0);
}

// Render 3D network using TEXTURE mode with FIXED orientation
static void DrawNetworkVis3D_Texture(NetworkVis* nv, Config* cfg, Camera3D cam, int showWeights, int* nodeVisible) {
    // Render all nodes using fixed textured quads
    for (int i = 0; i < VNODE_COUNT; i++) {
        if (!nv->nodes[i].visible) continue;
        if (nv->nodes[i].type == NODE_TYPE_WEIGHT && !showWeights) continue;
        
        VisNode* node = &nv->nodes[i];
        VisNode3DTransform* trans = &nv->transforms3d[i];
        NodeTextureCache* texCache = &nv->texture_caches[i];
        
        // Frustum culling
        float nodeRadius = fmaxf(trans->scale.x, trans->scale.y) * 0.7f;
        if (!IsInFrustum(trans->position, nodeRadius, cam, GetScreenWidth(), GetScreenHeight())) {
            continue;
        }
        nodeVisible[i] = 1;
        
        // Operations are rendered as spheres
        if (node->type == NODE_TYPE_OPERATION) {
            DrawSphere(trans->position, 0.5f, (Color){150, 150, 170, 255});
            continue;
        }
        
        // Skip if no grid data
        if (node->grid.rows <= 0 || node->grid.cols <= 0) continue;
        
        // Update texture cache
        if (texCache->dirty || !texCache->valid) {
            UpdateNodeTexture(texCache, &node->grid);
        }
        
        // Highlight picked node
        int isPicked = (nv->has_pick && nv->picked_node == i);
        
        // Draw as fixed textured quad (facing +Z, NOT facing camera)
        if (texCache->valid) {
            float width = trans->scale.x;
            float height = trans->scale.y;
            
            // Highlight with tint
            Color tint = isPicked ? (Color){255, 255, 200, 255} : WHITE;
            
            // Draw fixed quad - stays in 3D space, doesn't rotate to face camera
            DrawTexturedQuadFixed(texCache->texture, trans->position, width, height, tint);
            
            // Draw wireframe border
            DrawBoundingBox(trans->bbox, isPicked ? YELLOW : (Color){100, 100, 120, 150});
        } else {
            // Fallback: draw placeholder
            DrawCube(trans->position, 1.0f, 1.0f, 0.5f, (Color){80, 80, 80, 200});
            DrawCubeWires(trans->position, 1.0f, 1.0f, 0.5f, WHITE);
        }
    }
}

// Main 3D render function
static void DrawNetworkVis3D(NetworkVis* nv, Config* cfg) {
    Camera3D cam = nv->camera.camera;
    
    // Calculate camera distance for LOD
    float cameraDist = Vector3Distance(cam.position, (Vector3){0, 0, 24});
    
    // Skip weight matrices at distance (>50 units)
    int showWeights = nv->view.show_weights && (cameraDist < 50.0f);
    
    // Draw reference grid floor
    DrawGrid(50, 2.0f);
    
    // Draw axis indicators at origin
    DrawLine3D((Vector3){0, 0, 0}, (Vector3){5, 0, 0}, RED);    // X axis
    DrawLine3D((Vector3){0, 0, 0}, (Vector3){0, 5, 0}, GREEN);  // Y axis
    DrawLine3D((Vector3){0, 0, 0}, (Vector3){0, 0, 5}, BLUE);   // Z axis
    
    // Draw markers at key network positions
    DrawSphere((Vector3){0, 0, 0}, 0.3f, RED);     // Input
    DrawSphere((Vector3){0, 0, 24}, 0.3f, YELLOW); // Middle
    DrawSphere((Vector3){0, 0, 48}, 0.3f, GREEN);  // Output
    
    // Track visible nodes for edge rendering
    int nodeVisible[VNODE_COUNT] = {0};
    
    // Choose rendering mode
    if (nv->use_textures) {
        DrawNetworkVis3D_Texture(nv, cfg, cam, showWeights, nodeVisible);
    } else {
        DrawNetworkVis3D_Voxel(nv, cfg, cam, showWeights, nodeVisible);
    }
    
    // Draw edges/connections (only between visible nodes)
    for (int e = 0; e < nv->edge_count; e++) {
        DataFlowEdge* edge = &nv->edges[e];
        if (!nodeVisible[edge->from_node] || !nodeVisible[edge->to_node]) continue;
        
        Vector3 from = nv->transforms3d[edge->from_node].position;
        Vector3 to = nv->transforms3d[edge->to_node].position;
        
        // Offset slightly to avoid z-fighting
        from.z += 0.2f;
        to.z -= 0.2f;
        
        Draw3DEdge(from, to, edge->color, edge->animation_t, edge->active);
    }
}

// Update function for 3D mode - call before rendering
static void UpdateNetworkVis3D(NetworkVis* nv, RunState* state, TransformerWeights* weights, 
                                Config* cfg, int current_pos, Vector2 mousePos) {
    // Track if this is first time in 3D mode
    static int firstTime3D = 1;
    
    // Update data (only if position changed significantly, or first time)
    static int lastPos = -1;
    if (current_pos != lastPos || firstTime3D) {
        UpdateNetworkVisData(nv, state, weights, cfg, current_pos);
        lastPos = current_pos;
        
        // Mark all caches as dirty (data changed)
        for (int i = 0; i < VNODE_COUNT; i++) {
            nv->voxel_caches[i].dirty = 1;
            nv->texture_caches[i].dirty = 1;
        }
        firstTime3D = 0;
    }
    
    // Ensure 3D layout is computed
    LayoutNetworkVis3D(nv);
    
    // Update ray picking only when mouse moved (save performance)
    static Vector2 lastMousePos = {-1, -1};
    if (mousePos.x != lastMousePos.x || mousePos.y != lastMousePos.y) {
        UpdateRayPicking3D(nv, nv->camera.camera);
        lastMousePos = mousePos;
    }
}

// Handle 3D mode input
static void HandleNetworkVisInput3D(NetworkVis* nv, int n_layers) {
    // Layer navigation (same as 2D)
    if (IsKeyPressed(KEY_LEFT) && nv->view.current_layer > 0) {
        nv->view.current_layer--;
        // Mark caches dirty on layer change
        for (int i = 0; i < VNODE_COUNT; i++) {
            nv->voxel_caches[i].dirty = 1;
        }
    }
    if (IsKeyPressed(KEY_RIGHT) && nv->view.current_layer < n_layers - 1) {
        nv->view.current_layer++;
        for (int i = 0; i < VNODE_COUNT; i++) {
            nv->voxel_caches[i].dirty = 1;
        }
    }
    
    // Toggle weights display
    if (IsKeyPressed(KEY_W)) {
        nv->view.show_weights = !nv->view.show_weights;
        for (int i = 0; i < VNODE_COUNT; i++) {
            if (nv->nodes[i].type == NODE_TYPE_WEIGHT) {
                nv->nodes[i].visible = nv->view.show_weights;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Full Network Visualization Functions

// Compute min/max for a float array
static void ComputeMinMax(float *data, int count, float *out_min, float *out_max) {
    float min_v = FLT_MAX, max_v = -FLT_MAX;
    for (int i = 0; i < count; i++) {
        if (data[i] < min_v) min_v = data[i];
        if (data[i] > max_v) max_v = data[i];
    }
    *out_min = min_v;
    *out_max = max_v;
}

// Compute min/max for a quantized tensor (dequantized)
static void ComputeMinMaxQuantized(QuantizedTensor *qt, int count, float *out_min, float *out_max) {
    float min_v = FLT_MAX, max_v = -FLT_MAX;
    for (int i = 0; i < count; i++) {
        float val = qt->q[i] * qt->s[i / GS];
        if (val < min_v) min_v = val;
        if (val > max_v) max_v = val;
    }
    *out_min = min_v;
    *out_max = max_v;
}

// Get value from neuron grid
static float GetGridValue(NeuronGrid *grid, int row, int col) {
    int idx = row * grid->cols + col;
    if (grid->is_quantized && grid->qdata) {
        return grid->qdata->q[idx] * grid->qdata->s[idx / GS];
    } else if (grid->data) {
        return grid->data[idx];
    }
    return 0.0f;
}

// Initialize a visualization node
static void InitVisNode(VisNode *node, const char *name, const char *label, NodeType type) {
    node->name = name;
    node->label = label;
    node->type = type;
    node->grid = (NeuronGrid){0};
    node->pos = (Vector2){0, 0};
    node->width = 100;
    node->height = 50;
    node->visible = 1;
    node->collapsed = (type == NODE_TYPE_WEIGHT) ? 1 : 0;  // Weights collapsed by default
    node->highlight = 0;
}

// Initialize the full network visualization
static void InitNetworkVis(NetworkVis *nv, Config *cfg) {
    memset(nv, 0, sizeof(NetworkVis));
    nv->node_count = VNODE_COUNT;
    
    // Initialize layer view
    nv->view.current_layer = 0;
    nv->view.zoom = 1.0f;
    nv->view.pan = (Vector2){0, 0};
    nv->view.show_weights = 1;
    nv->view.show_activations = 1;
    nv->view.animation_speed = 1.0f;
    nv->view.animation_step = -1;
    nv->view.step_timer = 0;
    
    // Initialize activation nodes
    InitVisNode(&nv->nodes[VNODE_X], "x", "Hidden State", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_RMSNORM1], "RMS1", "RMSNorm", NODE_TYPE_OPERATION);
    InitVisNode(&nv->nodes[VNODE_XB], "xb", "Normalized", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_Q], "q", "Query", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_K], "k", "Key", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_V], "v", "Value", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_ATT], "att", "Attention", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_ATTV], "att*v", "Att Output", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_ATTN_OUT], "out", "Projection", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_RES1], "+", "Residual", NODE_TYPE_OPERATION);
    InitVisNode(&nv->nodes[VNODE_RMSNORM2], "RMS2", "RMSNorm", NODE_TYPE_OPERATION);
    InitVisNode(&nv->nodes[VNODE_XB2], "xb2", "Normalized", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_HB], "hb", "FFN Up", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_HB2], "hb2", "FFN Gate", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_SILU], "SiLU", "SiLU Gate", NODE_TYPE_OPERATION);
    InitVisNode(&nv->nodes[VNODE_FFN_OUT], "ffn", "FFN Output", NODE_TYPE_ACTIVATION);
    InitVisNode(&nv->nodes[VNODE_RES2], "+", "Residual", NODE_TYPE_OPERATION);
    InitVisNode(&nv->nodes[VNODE_XOUT], "x'", "Layer Output", NODE_TYPE_ACTIVATION);
    
    // Initialize weight nodes
    InitVisNode(&nv->nodes[VNODE_WQ], "Wq", "Query Weights", NODE_TYPE_WEIGHT);
    InitVisNode(&nv->nodes[VNODE_WK], "Wk", "Key Weights", NODE_TYPE_WEIGHT);
    InitVisNode(&nv->nodes[VNODE_WV], "Wv", "Value Weights", NODE_TYPE_WEIGHT);
    InitVisNode(&nv->nodes[VNODE_WO], "Wo", "Output Weights", NODE_TYPE_WEIGHT);
    InitVisNode(&nv->nodes[VNODE_W1], "W1", "FFN Up Weights", NODE_TYPE_WEIGHT);
    InitVisNode(&nv->nodes[VNODE_W2], "W2", "FFN Down Weights", NODE_TYPE_WEIGHT);
    InitVisNode(&nv->nodes[VNODE_W3], "W3", "FFN Gate Weights", NODE_TYPE_WEIGHT);
    
    // Initialize edges
    nv->edge_count = 0;
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_X, VNODE_RMSNORM1, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_RMSNORM1, VNODE_XB, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_XB, VNODE_WQ, (Color){255, 100, 100, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_WQ, VNODE_Q, (Color){255, 100, 100, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_XB, VNODE_WK, (Color){100, 255, 120, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_WK, VNODE_K, (Color){100, 255, 120, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_XB, VNODE_WV, (Color){100, 140, 255, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_WV, VNODE_V, (Color){100, 140, 255, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_Q, VNODE_ATT, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_K, VNODE_ATT, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_ATT, VNODE_ATTV, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_V, VNODE_ATTV, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_ATTV, VNODE_WO, (Color){180, 120, 255, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_WO, VNODE_ATTN_OUT, (Color){180, 120, 255, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_ATTN_OUT, VNODE_RES1, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_X, VNODE_RES1, (Color){150, 150, 170, 200}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_RES1, VNODE_RMSNORM2, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_RMSNORM2, VNODE_XB2, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_XB2, VNODE_W1, (Color){255, 160, 60, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_W1, VNODE_HB, (Color){255, 160, 60, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_XB2, VNODE_W3, (Color){255, 220, 80, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_W3, VNODE_HB2, (Color){255, 220, 80, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_HB, VNODE_SILU, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_HB2, VNODE_SILU, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_SILU, VNODE_W2, (Color){60, 220, 220, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_W2, VNODE_FFN_OUT, (Color){60, 220, 220, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_FFN_OUT, VNODE_RES2, (Color){150, 150, 170, 255}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_RES1, VNODE_RES2, (Color){150, 150, 170, 200}, 0, 0};
    nv->edges[nv->edge_count++] = (DataFlowEdge){VNODE_RES2, VNODE_XOUT, (Color){150, 150, 170, 255}, 0, 0};
    
    nv->initialized = 1;
    nv->hover_node = -1;
    nv->has_hover = 0;

    // === Initialize 3D components ===
    // Initialize voxel caches
    for (int i = 0; i < VNODE_COUNT; i++) {
        nv->voxel_caches[i].instances = NULL;
        nv->voxel_caches[i].instance_count = 0;
        nv->voxel_caches[i].capacity = 0;
        nv->voxel_caches[i].dirty = 1;  // Start dirty to trigger initial build
    }
    
    // Initialize texture caches
    for (int i = 0; i < VNODE_COUNT; i++) {
        nv->texture_caches[i].valid = 0;
        nv->texture_caches[i].dirty = 1;
    }

    // Initialize 3D camera (positioned to see the whole network)
    Vector3 start_pos = {25.0f, 20.0f, -20.0f};  // Side view
    Vector3 look_at = {0.0f, 0.0f, 24.0f};       // Look at center of network
    InitFreeFlightCamera(&nv->camera, start_pos, look_at);

    // Start in 2D mode by default, can toggle to 3D with a key
    nv->use_3d_mode = 0;
    nv->use_textures = 1;  // Use texture rendering by default

    // Initialize ray picking state
    nv->picked_node = -1;
    nv->has_pick = 0;
    
    // Compute initial 3D layout
    LayoutNetworkVis3D(nv);
}

// Update node data pointers for current layer
static void UpdateNetworkVisData(NetworkVis *nv, RunState *state, TransformerWeights *weights, 
                                  Config *cfg, int current_pos) {
    int layer = nv->view.current_layer;
    int all_heads_dim = cfg->n_heads * cfg->head_dim;
    int kv_dim = cfg->n_kv_heads * cfg->head_dim;
    int seq_len = current_pos + 1;
    if (seq_len < 1) seq_len = 1;
    if (seq_len > cfg->seq_len) seq_len = cfg->seq_len;
    
    // Activation nodes - point to RunState data
    nv->nodes[VNODE_X].grid = (NeuronGrid){
        .data = state->x, .rows = 32, .cols = cfg->dim / 32, .is_quantized = 0
    };
    nv->nodes[VNODE_XB].grid = (NeuronGrid){
        .data = state->xb, .rows = 32, .cols = all_heads_dim / 32, .is_quantized = 0
    };
    nv->nodes[VNODE_Q].grid = (NeuronGrid){
        .data = state->q, .rows = cfg->n_heads, .cols = cfg->head_dim, .is_quantized = 0
    };
    nv->nodes[VNODE_K].grid = (NeuronGrid){
        .data = state->k, .rows = cfg->n_kv_heads, .cols = cfg->head_dim, .is_quantized = 0
    };
    nv->nodes[VNODE_V].grid = (NeuronGrid){
        .data = state->v, .rows = cfg->n_kv_heads, .cols = cfg->head_dim, .is_quantized = 0
    };
    nv->nodes[VNODE_ATT].grid = (NeuronGrid){
        .data = state->att, .rows = cfg->n_heads, .cols = seq_len, .is_quantized = 0
    };
    nv->nodes[VNODE_ATTV].grid = (NeuronGrid){
        .data = state->xb, .rows = cfg->n_heads, .cols = cfg->head_dim, .is_quantized = 0
    };
    nv->nodes[VNODE_ATTN_OUT].grid = (NeuronGrid){
        .data = state->xb, .rows = 32, .cols = cfg->dim / 32, .is_quantized = 0
    };
    nv->nodes[VNODE_XB2].grid = (NeuronGrid){
        .data = state->xb, .rows = 32, .cols = cfg->dim / 32, .is_quantized = 0
    };
    nv->nodes[VNODE_HB].grid = (NeuronGrid){
        .data = state->hb, .rows = 64, .cols = cfg->hidden_dim / 64, .is_quantized = 0
    };
    nv->nodes[VNODE_HB2].grid = (NeuronGrid){
        .data = state->hb2, .rows = 64, .cols = cfg->hidden_dim / 64, .is_quantized = 0
    };
    nv->nodes[VNODE_FFN_OUT].grid = (NeuronGrid){
        .data = state->xb, .rows = 32, .cols = cfg->dim / 32, .is_quantized = 0
    };
    nv->nodes[VNODE_XOUT].grid = (NeuronGrid){
        .data = state->x, .rows = 32, .cols = cfg->dim / 32, .is_quantized = 0
    };
    
    // Weight nodes - point to quantized weights for current layer
    nv->nodes[VNODE_WQ].grid = (NeuronGrid){
        .qdata = &weights->wq[layer], .rows = cfg->dim, .cols = all_heads_dim, .is_quantized = 1
    };
    nv->nodes[VNODE_WK].grid = (NeuronGrid){
        .qdata = &weights->wk[layer], .rows = cfg->dim, .cols = kv_dim, .is_quantized = 1
    };
    nv->nodes[VNODE_WV].grid = (NeuronGrid){
        .qdata = &weights->wv[layer], .rows = cfg->dim, .cols = kv_dim, .is_quantized = 1
    };
    nv->nodes[VNODE_WO].grid = (NeuronGrid){
        .qdata = &weights->wo[layer], .rows = all_heads_dim, .cols = cfg->dim, .is_quantized = 1
    };
    nv->nodes[VNODE_W1].grid = (NeuronGrid){
        .qdata = &weights->w1[layer], .rows = cfg->dim, .cols = cfg->hidden_dim, .is_quantized = 1
    };
    nv->nodes[VNODE_W2].grid = (NeuronGrid){
        .qdata = &weights->w2[layer], .rows = cfg->hidden_dim, .cols = cfg->dim, .is_quantized = 1
    };
    nv->nodes[VNODE_W3].grid = (NeuronGrid){
        .qdata = &weights->w3[layer], .rows = cfg->dim, .cols = cfg->hidden_dim, .is_quantized = 1
    };
    
    // Animate data flow - advance step on each update
    nv->view.step_timer += 0.5f;  // Advance animation
    if (nv->view.step_timer >= 1.0f) {
        nv->view.step_timer = 0;
        nv->view.animation_step = (nv->view.animation_step + 1) % nv->edge_count;
    }
    
    // Update edge animations
    for (int i = 0; i < nv->edge_count; i++) {
        DataFlowEdge *edge = &nv->edges[i];
        if (i == nv->view.animation_step) {
            edge->active = 1;
            edge->animation_t += 0.15f * nv->view.animation_speed;
            if (edge->animation_t > 1.0f) edge->animation_t = 0;
            
            // Highlight connected nodes
            nv->nodes[edge->from_node].highlight = 2;
            nv->nodes[edge->to_node].highlight = 1;
        } else {
            edge->active = 0;
            edge->animation_t = 0;
            // Fade out highlights
            if (nv->nodes[i].highlight > 0) nv->nodes[i].highlight--;
        }
    }
    
    // Compute min/max for activation nodes
    for (int i = 0; i < VNODE_COUNT; i++) {
        NeuronGrid *g = &nv->nodes[i].grid;
        if (g->rows > 0 && g->cols > 0) {
            int count = g->rows * g->cols;
            if (g->is_quantized && g->qdata) {
                ComputeMinMaxQuantized(g->qdata, count, &g->min_val, &g->max_val);
            } else if (g->data) {
                ComputeMinMax(g->data, count, &g->min_val, &g->max_val);
            }
            g->values_computed = 1;
        }
    }
}

// Layout nodes vertically
static void LayoutNetworkVis(NetworkVis *nv, Rectangle area, float zoom) {
    float padding = 20 * zoom;
    float node_spacing = 40 * zoom;
    float y = area.y + padding;
    float centerX = area.x + area.width / 2;
    
    // Attention block layout (left column)
    float leftX = area.x + area.width * 0.25f;
    float rightX = area.x + area.width * 0.75f;
    
    // Row 1: x -> RMSNorm -> xb
    nv->nodes[VNODE_X].pos = (Vector2){leftX - 120 * zoom, y};
    nv->nodes[VNODE_RMSNORM1].pos = (Vector2){leftX, y};
    nv->nodes[VNODE_XB].pos = (Vector2){leftX + 120 * zoom, y};
    y += 80 * zoom + node_spacing;
    
    // Row 2: Weight matrices (Wq, Wk, Wv)
    if (nv->view.show_weights && !nv->nodes[VNODE_WQ].collapsed) {
        nv->nodes[VNODE_WQ].pos = (Vector2){leftX - 100 * zoom, y};
        nv->nodes[VNODE_WK].pos = (Vector2){leftX, y};
        nv->nodes[VNODE_WV].pos = (Vector2){leftX + 100 * zoom, y};
        y += 100 * zoom + node_spacing;
    }
    
    // Row 3: Q, K, V vectors
    nv->nodes[VNODE_Q].pos = (Vector2){leftX - 100 * zoom, y};
    nv->nodes[VNODE_K].pos = (Vector2){leftX, y};
    nv->nodes[VNODE_V].pos = (Vector2){leftX + 100 * zoom, y};
    y += 60 * zoom + node_spacing;
    
    // Row 4: Attention scores
    nv->nodes[VNODE_ATT].pos = (Vector2){centerX, y};
    y += 80 * zoom + node_spacing;
    
    // Row 5: Att*V output
    nv->nodes[VNODE_ATTV].pos = (Vector2){leftX, y};
    if (nv->view.show_weights && !nv->nodes[VNODE_WO].collapsed) {
        nv->nodes[VNODE_WO].pos = (Vector2){leftX + 120 * zoom, y};
    }
    y += 80 * zoom + node_spacing;
    
    // Row 6: Attention output + Residual
    nv->nodes[VNODE_ATTN_OUT].pos = (Vector2){leftX, y};
    nv->nodes[VNODE_RES1].pos = (Vector2){leftX + 120 * zoom, y};
    y += 60 * zoom + node_spacing;
    
    // Row 7: RMSNorm2 -> xb2
    nv->nodes[VNODE_RMSNORM2].pos = (Vector2){rightX - 60 * zoom, y};
    nv->nodes[VNODE_XB2].pos = (Vector2){rightX + 60 * zoom, y};
    y += 60 * zoom + node_spacing;
    
    // Row 8: FFN weight matrices (W1, W3)
    if (nv->view.show_weights && !nv->nodes[VNODE_W1].collapsed) {
        nv->nodes[VNODE_W1].pos = (Vector2){rightX - 80 * zoom, y};
        nv->nodes[VNODE_W3].pos = (Vector2){rightX + 80 * zoom, y};
        y += 100 * zoom + node_spacing;
    }
    
    // Row 9: hb, hb2
    nv->nodes[VNODE_HB].pos = (Vector2){rightX - 80 * zoom, y};
    nv->nodes[VNODE_HB2].pos = (Vector2){rightX + 80 * zoom, y};
    y += 80 * zoom + node_spacing;
    
    // Row 10: SiLU gate
    nv->nodes[VNODE_SILU].pos = (Vector2){rightX, y};
    y += 50 * zoom + node_spacing;
    
    // Row 11: W2 and FFN output
    if (nv->view.show_weights && !nv->nodes[VNODE_W2].collapsed) {
        nv->nodes[VNODE_W2].pos = (Vector2){rightX - 60 * zoom, y};
    }
    nv->nodes[VNODE_FFN_OUT].pos = (Vector2){rightX + 60 * zoom, y};
    y += 80 * zoom + node_spacing;
    
    // Row 12: Residual 2 and output
    nv->nodes[VNODE_RES2].pos = (Vector2){rightX - 60 * zoom, y};
    nv->nodes[VNODE_XOUT].pos = (Vector2){rightX + 60 * zoom, y};
}

// Draw a single neuron grid
static void DrawNeuronGrid(NeuronGrid *grid, Vector2 pos, float cell_size, 
                           int max_cells, int *out_hover_row, int *out_hover_col,
                           float *out_hover_val, Vector2 mouse) {
    if (grid->rows <= 0 || grid->cols <= 0) return;
    
    // Limit cells to draw for performance
    int draw_rows = grid->rows;
    int draw_cols = grid->cols;
    if (max_cells > 0) {
        if (draw_rows * draw_cols > max_cells) {
            // Subsample
            float ratio = sqrtf((float)max_cells / (draw_rows * draw_cols));
            draw_rows = (int)(draw_rows * ratio);
            draw_cols = (int)(draw_cols * ratio);
            if (draw_rows < 1) draw_rows = 1;
            if (draw_cols < 1) draw_cols = 1;
        }
    }
    
    float range = grid->max_val - grid->min_val;
    if (range < 1e-6f) range = 1.0f;
    
    int row_step = grid->rows / draw_rows;
    int col_step = grid->cols / draw_cols;
    if (row_step < 1) row_step = 1;
    if (col_step < 1) col_step = 1;
    
    for (int dr = 0; dr < draw_rows; dr++) {
        for (int dc = 0; dc < draw_cols; dc++) {
            int r = dr * row_step;
            int c = dc * col_step;
            float val = GetGridValue(grid, r, c);
            float t = (val - grid->min_val) / range;
            Color color = FloatToColorInferno(t);
            
            float x = pos.x + dc * cell_size;
            float y = pos.y + dr * cell_size;
            DrawRectangle((int)x, (int)y, (int)(cell_size - 1), (int)(cell_size - 1), color);
            
            // Check hover
            if (mouse.x >= x && mouse.x < x + cell_size &&
                mouse.y >= y && mouse.y < y + cell_size) {
                *out_hover_row = r;
                *out_hover_col = c;
                *out_hover_val = val;
            }
        }
    }
}

// Draw a node box with label
static void DrawNodeBox(VisNode *node, Vector2 pos, float width, float height, 
                        float cell_size, int max_cells, Vector2 mouse,
                        int *out_hover_row, int *out_hover_col, float *out_hover_val) {
    Rectangle rect = {pos.x - width / 2, pos.y - height / 2, width, height};
    
    // Background
    Color bg = (node->type == NODE_TYPE_WEIGHT) ? (Color){30, 35, 45, 255} :
               (node->type == NODE_TYPE_OPERATION) ? (Color){40, 40, 50, 255} :
               (Color){35, 35, 42, 255};
    
    DrawRectangleRounded(rect, 0.1f, 8, bg);
    
    // Highlight border if active
    Color border = node->highlight > 0 ? (Color){255, 200, 100, 255} : (Color){70, 70, 85, 255};
    DrawRectangleRoundedLines(rect, 0.1f, 8, border);
    
    // Draw grid inside
    if (node->grid.rows > 0 && node->grid.cols > 0 && node->type != NODE_TYPE_OPERATION) {
        float inner_padding = (float)UI_SCALE_I(4);
        Vector2 grid_pos = {rect.x + inner_padding, rect.y + inner_padding + UI_SCALE_I(14)};
        float grid_w = width - inner_padding * 2;
        float grid_h = height - inner_padding * 2 - UI_SCALE_I(18);
        
        // Calculate cell size to fit
        int draw_rows = node->grid.rows;
        int draw_cols = node->grid.cols;
        if (max_cells > 0 && draw_rows * draw_cols > max_cells) {
            float ratio = sqrtf((float)max_cells / (draw_rows * draw_cols));
            draw_rows = (int)(draw_rows * ratio);
            draw_cols = (int)(draw_cols * ratio);
            if (draw_rows < 1) draw_rows = 1;
            if (draw_cols < 1) draw_cols = 1;
        }
        
        float actual_cell = fminf(grid_w / draw_cols, grid_h / draw_rows);
        if (actual_cell < 1) actual_cell = 1;
        
        DrawNeuronGrid(&node->grid, grid_pos, actual_cell, max_cells, 
                       out_hover_row, out_hover_col, out_hover_val, mouse);
    }
    
    // Label
    int label_w = MeasureTextUI(node->name, 12);
    DrawTextUI(node->name, (int)(rect.x + (width - label_w) / 2), (int)(rect.y + UI_SCALE_I(2)), 12, TEXT_COLOR);

    // Size info
    if (node->grid.rows > 0 && node->grid.cols > 0) {
        char size_str[32];
        sprintf(size_str, "%dx%d", node->grid.rows, node->grid.cols);
        int size_w = MeasureTextUI(size_str, 10);
        DrawTextUI(size_str, (int)(rect.x + width - size_w - UI_SCALE_I(4)), (int)(rect.y + height - UI_SCALE_I(14)), 10, PLACEHOLDER_COLOR);
    }
}

// Draw edge between two nodes
static void DrawNodeEdge(VisNode *from, VisNode *to, Color color, float animation_t, int active) {
    Vector2 start = {from->pos.x, from->pos.y + from->height / 2};
    Vector2 end = {to->pos.x, to->pos.y - to->height / 2};
    
    // Draw line - thicker and brighter when active
    float thickness = active ? 3.0f : 1.5f;
    Color draw_color = color;
    if (active) {
        // Brighten the color when active
        draw_color.r = (unsigned char)fminf(255, color.r * 1.3f);
        draw_color.g = (unsigned char)fminf(255, color.g * 1.3f);
        draw_color.b = (unsigned char)fminf(255, color.b * 1.3f);
    }
    DrawLineEx(start, end, thickness, draw_color);
    
    // Animation particle when active
    if (active && animation_t > 0 && animation_t < 1) {
        // Draw multiple particles for a trail effect
        for (int p = 0; p < 3; p++) {
            float t = animation_t - p * 0.1f;
            if (t < 0) t += 1.0f;
            if (t > 1) t -= 1.0f;
            
            Vector2 particle = {
                start.x + (end.x - start.x) * t,
                start.y + (end.y - start.y) * t
            };
            float radius = 5 - p * 1.5f;
            unsigned char alpha = 255 - p * 60;
            DrawCircle((int)particle.x, (int)particle.y, radius, 
                      (Color){255, 255, 200, alpha});
        }
    }
}

// Draw the full network visualization
static void DrawNetworkVis(NetworkVis *nv, Rectangle area, Vector2 mouse, Config *cfg) {
    // Apply zoom and pan
    float zoom = nv->view.zoom;
    Vector2 pan = nv->view.pan;
    
    // Adjust area with pan
    Rectangle view_area = {
        area.x + pan.x,
        area.y + pan.y,
        area.width,
        area.height * 3  // Extended for scrolling
    };
    
    // Layout nodes
    LayoutNetworkVis(nv, view_area, zoom);
    
    // Draw edges first (behind nodes)
    for (int i = 0; i < nv->edge_count; i++) {
        DataFlowEdge *edge = &nv->edges[i];
        VisNode *from = &nv->nodes[edge->from_node];
        VisNode *to = &nv->nodes[edge->to_node];
        if (from->visible && to->visible) {
            DrawNodeEdge(from, to, edge->color, edge->animation_t, edge->active);
        }
    }
    
    // Draw nodes
    nv->has_hover = 0;
    for (int i = 0; i < VNODE_COUNT; i++) {
        VisNode *node = &nv->nodes[i];
        if (!node->visible) continue;
        
        // Skip collapsed weight nodes
        if (node->type == NODE_TYPE_WEIGHT && node->collapsed && !nv->view.show_weights) continue;
        
        // Determine node size based on type and zoom
        float width, height;
        int max_cells;
        if (node->type == NODE_TYPE_WEIGHT) {
            width = node->collapsed ? 60 * zoom : 120 * zoom;
            height = node->collapsed ? 30 * zoom : 100 * zoom;
            max_cells = node->collapsed ? 100 : 2000;
        } else if (node->type == NODE_TYPE_OPERATION) {
            width = 50 * zoom;
            height = 30 * zoom;
            max_cells = 0;
        } else {
            width = 100 * zoom;
            height = 70 * zoom;
            max_cells = 1000;
        }
        
        node->width = width;
        node->height = height;
        
        // Calculate cell size
        float cell_size = zoom * 3;
        if (cell_size < 1) cell_size = 1;
        
        int hover_row = -1, hover_col = -1;
        float hover_val = 0;
        
        DrawNodeBox(node, node->pos, width, height, cell_size, max_cells, 
                    mouse, &hover_row, &hover_col, &hover_val);
        
        // Check if mouse is hovering over this node
        Rectangle rect = {node->pos.x - width / 2, node->pos.y - height / 2, width, height};
        if (CheckCollisionPointRec(mouse, rect)) {
            nv->hover_node = i;
            nv->hover_row = hover_row;
            nv->hover_col = hover_col;
            nv->hover_value = hover_val;
            nv->has_hover = 1;
        }
    }
}

// Draw layer navigation UI
static void DrawLayerNav(NetworkVis *nv, Rectangle area, int n_layers) {
    int nav_height = UI_SCALE_I(40);
    Rectangle nav_rect = {area.x, area.y, area.width, nav_height};
    
    DrawRectangle((int)nav_rect.x, (int)nav_rect.y, (int)nav_rect.width, nav_height, TITLE_BG_COLOR);
    
    // Layer indicator
    char layer_text[64];
    sprintf(layer_text, "Layer %d / %d", nv->view.current_layer + 1, n_layers);
    int text_w = MeasureTextUI(layer_text, 18);
    DrawTextUI(layer_text, (int)(nav_rect.x + (nav_rect.width - text_w) / 2), (int)(nav_rect.y + UI_SCALE_I(10)), 18, TEXT_COLOR);

    // Navigation arrows
    DrawTextUI("<", (int)(nav_rect.x + UI_SCALE_I(20)), (int)(nav_rect.y + UI_SCALE_I(8)), 24,
             nv->view.current_layer > 0 ? ACCENT_COLOR : PLACEHOLDER_COLOR);
    DrawTextUI(">", (int)(nav_rect.x + nav_rect.width - UI_SCALE_I(35)), (int)(nav_rect.y + UI_SCALE_I(8)), 24,
             nv->view.current_layer < n_layers - 1 ? ACCENT_COLOR : PLACEHOLDER_COLOR);

    // Controls hint
    DrawTextUI("[<-/->] Navigate  [W] Weights  [+/-] Zoom  [R] Reset",
             (int)(nav_rect.x + UI_SCALE_I(10)), (int)(nav_rect.y + nav_height - UI_SCALE_I(14)), 11, PLACEHOLDER_COLOR);
}

// Draw hover tooltip
static void DrawHoverTooltip(NetworkVis *nv, Vector2 mouse) {
    if (!nv->has_hover || nv->hover_node < 0) return;
    
    VisNode *node = &nv->nodes[nv->hover_node];
    
    char tip[128];
    if (nv->hover_row >= 0 && nv->hover_col >= 0) {
        sprintf(tip, "%s [%d,%d] = %.4f", node->name, nv->hover_row, nv->hover_col, nv->hover_value);
    } else {
        sprintf(tip, "%s (%s)", node->name, node->label);
    }

    int tip_w = MeasureTextUI(tip, 14) + UI_SCALE_I(20);
    int tip_h = UI_SCALE_I(28);
    int tip_x = (int)mouse.x + UI_SCALE_I(15);
    int tip_y = (int)mouse.y + UI_SCALE_I(15);

    DrawRectangle(tip_x, tip_y, tip_w, tip_h, (Color){20, 20, 28, 240});
    DrawRectangleLines(tip_x, tip_y, tip_w, tip_h, (Color){80, 80, 100, 255});
    DrawTextUI(tip, tip_x + UI_SCALE_I(10), tip_y + UI_SCALE_I(7), 14, TEXT_COLOR);
}

// ----------------------------------------------------------------------------
// 3D Camera Control Functions

// Initialize free-flight camera
static void InitFreeFlightCamera(FreeFlightCamera* cam, Vector3 start_pos, Vector3 look_at) {
    cam->camera.position = start_pos;
    cam->camera.target = look_at;
    cam->camera.up = (Vector3){0.0f, 1.0f, 0.0f};
    cam->camera.fovy = 60.0f;
    cam->camera.projection = CAMERA_PERSPECTIVE;

    cam->velocity.x = 0.0f;
    cam->velocity.y = 0.0f;
    cam->velocity.z = 0.0f;
    cam->move_speed = 10.0f;
    cam->mouse_sensitivity = 0.003f;
    cam->locked = 0;

    // Calculate initial yaw/pitch from look direction
    Vector3 forward;
    forward.x = look_at.x - start_pos.x;
    forward.y = look_at.y - start_pos.y;
    forward.z = look_at.z - start_pos.z;

    float len = sqrtf(forward.x*forward.x + forward.y*forward.y + forward.z*forward.z);
    if (len > 0.0001f) {
        forward.x /= len;
        forward.y /= len;
        forward.z /= len;
    }

    cam->yaw = atan2f(forward.z, forward.x);
    cam->pitch = asinf(forward.y);
}

// Update free-flight camera with WASD + mouse controls
static void UpdateFreeFlightCamera(FreeFlightCamera* cam, float dt) {
    // Toggle mouse lock with right-click
    if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
        cam->locked = !cam->locked;
        if (cam->locked) {
            DisableCursor();
        } else {
            EnableCursor();
        }
    }

    // Mouse look (when locked)
    if (cam->locked) {
        Vector2 delta = GetMouseDelta();
        cam->yaw += delta.x * cam->mouse_sensitivity;
        cam->pitch -= delta.y * cam->mouse_sensitivity;

        // Clamp pitch to avoid gimbal lock
        const float MAX_PITCH = PI / 2.0f - 0.1f;
        if (cam->pitch > MAX_PITCH) cam->pitch = MAX_PITCH;
        if (cam->pitch < -MAX_PITCH) cam->pitch = -MAX_PITCH;
    }

    // Calculate forward and right vectors
    Vector3 forward = {
        cosf(cam->yaw) * cosf(cam->pitch),
        sinf(cam->pitch),
        sinf(cam->yaw) * cosf(cam->pitch)
    };
    Vector3 right = Vector3CrossProduct(forward, (Vector3){0, 1, 0});
    right = Vector3Normalize(right);

    // WASD + Space + Ctrl movement
    Vector3 move_input = {0, 0, 0};
    if (IsKeyDown(KEY_W)) {
        move_input = Vector3Add(move_input, forward);
    }
    if (IsKeyDown(KEY_S)) {
        move_input = Vector3Subtract(move_input, forward);
    }
    if (IsKeyDown(KEY_D)) {
        move_input = Vector3Add(move_input, right);
    }
    if (IsKeyDown(KEY_A)) {
        move_input = Vector3Subtract(move_input, right);
    }
    if (IsKeyDown(KEY_SPACE)) {
        move_input.y += 1.0f;
    }
    if (IsKeyDown(KEY_LEFT_CONTROL)) {
        move_input.y -= 1.0f;
    }

    // Apply velocity
    if (Vector3Length(move_input) > 0.01f) {
        move_input = Vector3Normalize(move_input);
        cam->velocity = Vector3Scale(move_input, cam->move_speed);
    } else {
        cam->velocity = (Vector3){0, 0, 0};
    }

    // Update position and target
    cam->camera.position = Vector3Add(cam->camera.position,
                                      Vector3Scale(cam->velocity, dt));
    cam->camera.target = Vector3Add(cam->camera.position, forward);
}

// Handle network vis input
static void HandleNetworkVisInput(NetworkVis *nv, int n_layers, Rectangle area, Vector2 mouse) {
    // Layer navigation
    if (IsKeyPressed(KEY_LEFT) && nv->view.current_layer > 0) {
        nv->view.current_layer--;
    }
    if (IsKeyPressed(KEY_RIGHT) && nv->view.current_layer < n_layers - 1) {
        nv->view.current_layer++;
    }
    
    // Toggle weights display
    if (IsKeyPressed(KEY_W)) {
        nv->view.show_weights = !nv->view.show_weights;
    }
    
    // Zoom
    if (CheckCollisionPointRec(mouse, area)) {
        float wheel = GetMouseWheelMove();
        if (wheel != 0) {
            nv->view.zoom *= (1.0f + wheel * 0.1f);
            if (nv->view.zoom < 0.3f) nv->view.zoom = 0.3f;
            if (nv->view.zoom > 3.0f) nv->view.zoom = 3.0f;
        }
        
        // Pan with right mouse button
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Vector2 delta = GetMouseDelta();
            nv->view.pan.x += delta.x;
            nv->view.pan.y += delta.y;
        }
    }
    
    // Keyboard zoom
    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
        nv->view.zoom *= 1.2f;
        if (nv->view.zoom > 3.0f) nv->view.zoom = 3.0f;
    }
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
        nv->view.zoom /= 1.2f;
        if (nv->view.zoom < 0.3f) nv->view.zoom = 0.3f;
    }
    
    // Reset view
    if (IsKeyPressed(KEY_R)) {
        nv->view.zoom = 1.0f;
        nv->view.pan = (Vector2){0, 0};
    }
    
    // Toggle collapsed state on click
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && nv->has_hover) {
        VisNode *node = &nv->nodes[nv->hover_node];
        if (node->type == NODE_TYPE_WEIGHT) {
            node->collapsed = !node->collapsed;
        }
    }
}

// ----------------------------------------------------------------------------
// Helper functions

static int GetPreviousCodepointOffset(const char *text, int bytePos) {
    if (bytePos <= 0) return 0;

    int offset = 1;
    while (bytePos - offset > 0 && ((unsigned char)text[bytePos - offset] & 0xC0) == 0x80) {
        offset++;
    }

    return offset;
}

static void AppendCodepointUTF8(char *buffer, int *length, int max_len, int codepoint) {
    int byteCount = 0;
    const char *utf8 = CodepointToUTF8(codepoint, &byteCount);
    if (byteCount <= 0) return;
    if ((*length) + byteCount >= max_len) return;

    for (int i = 0; i < byteCount; i++) {
        buffer[(*length)++] = utf8[i];
    }
    buffer[*length] = '\0';
}

static float MeasureCodepointWidth(Font font, int codepoint, float fontSize, float spacing) {
    int byteCount = 0;
    const char *utf8 = CodepointToUTF8(codepoint, &byteCount);
    char buf[8] = {0};
    if (byteCount <= 0 || byteCount > (int)(sizeof(buf) - 1)) return 0.0f;
    memcpy(buf, utf8, byteCount);
    buf[byteCount] = '\0';
    return MeasureTextEx(font, buf, fontSize, spacing).x;
}

#if defined(_WIN32)
static void UpdateImeWindowPosition(int x, int y) {
    HWND hwnd = (HWND)GetWindowHandle();
    if (!hwnd) return;

    HIMC imc = ImmGetContext(hwnd);
    if (!imc) return;

    COMPOSITIONFORM cf = {0};
    cf.dwStyle = CFS_POINT;
    cf.ptCurrentPos.x = x;
    cf.ptCurrentPos.y = y;
    ImmSetCompositionWindow(imc, &cf);

    CANDIDATEFORM cand = {0};
    cand.dwStyle = CFS_CANDIDATEPOS;
    cand.ptCurrentPos.x = x;
    cand.ptCurrentPos.y = y;
    ImmSetCandidateWindow(imc, &cand);

    ImmReleaseContext(hwnd, imc);
}
#else
static void UpdateImeWindowPosition(int x, int y) {
    (void)x;
    (void)y;
}
#endif

static Font GetUIFont(void) {
    return (g_ui_font.texture.id != 0) ? g_ui_font : GetFontDefault();
}

static int MeasureTextUI(const char *text, int fontSize) {
    Font font = GetUIFont();
    int scaledSize = fontSize * UI_SCALE;
    return (int)(MeasureTextEx(font, text, (float)scaledSize, 1.0f).x + 0.5f);
}

static void DrawTextUI(const char *text, float x, float y, int fontSize, Color color) {
    Font font = GetUIFont();
    int scaledSize = fontSize * UI_SCALE;
    DrawTextEx(font, text, (Vector2){x, y}, (float)scaledSize, 1.0f, color);
}

static Font LoadChatFont(int fontSize) {
    int codepoints[23000];
    int codepointCount = 0;

    for (int i = 32; i < 127; i++) codepoints[codepointCount++] = i;
    for (int i = 0x4E00; i <= 0x9FFF; i++) codepoints[codepointCount++] = i;
    for (int i = 0x3000; i <= 0x303F; i++) codepoints[codepointCount++] = i;
    for (int i = 0x3040; i <= 0x309F; i++) codepoints[codepointCount++] = i;
    for (int i = 0x30A0; i <= 0x30FF; i++) codepoints[codepointCount++] = i;
    for (int i = 0xFF00; i <= 0xFFEF; i++) codepoints[codepointCount++] = i;

    char exeFontPath[1024];
    char exeFontPathUp1[1024];
    char exeFontPathUp2[1024];
    char exeAssetsPath[1024];
    char exeAssetsPathUp1[1024];
    char exeAssetsPathUp2[1024];
    char exeFontPathOtf[1024];
    char exeFontPathUp1Otf[1024];
    char exeFontPathUp2Otf[1024];

    const char *exeDir = GetApplicationDirectory();
    snprintf(exeFontPath, sizeof(exeFontPath), "%sNotoSansCJK-Regular.ttc", exeDir);
    snprintf(exeFontPathUp1, sizeof(exeFontPathUp1), "%s../NotoSansCJK-Regular.ttc", exeDir);
    snprintf(exeFontPathUp2, sizeof(exeFontPathUp2), "%s../../NotoSansCJK-Regular.ttc", exeDir);
    snprintf(exeAssetsPath, sizeof(exeAssetsPath), "%sassets/fonts/NotoSansCJK-Regular.ttc", exeDir);
    snprintf(exeAssetsPathUp1, sizeof(exeAssetsPathUp1), "%s../assets/fonts/NotoSansCJK-Regular.ttc", exeDir);
    snprintf(exeAssetsPathUp2, sizeof(exeAssetsPathUp2), "%s../../assets/fonts/NotoSansCJK-Regular.ttc", exeDir);
    snprintf(exeFontPathOtf, sizeof(exeFontPathOtf), "%sNotoSansCJK-Regular.otf", exeDir);
    snprintf(exeFontPathUp1Otf, sizeof(exeFontPathUp1Otf), "%s../NotoSansCJK-Regular.otf", exeDir);
    snprintf(exeFontPathUp2Otf, sizeof(exeFontPathUp2Otf), "%s../../NotoSansCJK-Regular.otf", exeDir);

    const char *fontPaths[] = {
        "NotoSansCJK-Regular.ttc",
        "NotoSansCJK-Regular.otf",
        "assets/fonts/NotoSansCJK-Regular.ttc",
        "assets/fonts/NotoSansCJK-Regular.otf",
        exeFontPath,
        exeFontPathUp1,
        exeFontPathUp2,
        exeAssetsPath,
        exeAssetsPathUp1,
        exeAssetsPathUp2,
        exeFontPathOtf,
        exeFontPathUp1Otf,
        exeFontPathUp2Otf,
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        NULL
    };

    Font font = {0};
    Font defaultFont = GetFontDefault();
    const char *loadedPath = NULL;
    for (int i = 0; fontPaths[i] != NULL; i++) {
        if (FileExists(fontPaths[i])) {
            TraceLog(LOG_INFO, "FONT: Trying %s", fontPaths[i]);
            Font loaded = LoadFontEx(fontPaths[i], fontSize, codepoints, codepointCount);
            if (loaded.texture.id != 0 &&
                !(loaded.texture.id == defaultFont.texture.id && loaded.glyphCount == defaultFont.glyphCount)) {
                font = loaded;
                loadedPath = fontPaths[i];
                break;
            }

            if (loaded.texture.id != 0 && loaded.texture.id != defaultFont.texture.id) {
                UnloadFont(loaded);
            }
            TraceLog(LOG_WARNING, "FONT: Failed to load %s", fontPaths[i]);
        }
    }

    if (loadedPath) {
        TraceLog(LOG_INFO, "FONT: Loaded %s (%d glyphs)", loadedPath, font.glyphCount);
    } else {
        TraceLog(LOG_WARNING, "FONT: Could not load CJK font, using default");
        font = defaultFont;
    }
    return font;
}

int MeasureTextWrapped(Font font, const char *text, int fontSize, int maxWidth, int *outHeight) {
    if (text == NULL || text[0] == '\0') {
        *outHeight = LINE_HEIGHT;
        return 1;
    }
    
    int lines = 1;
    float lineWidth = 0.0f;
    float lastSpaceWidth = -1.0f;

    const char *ptr = text;
    while (*ptr != '\0') {
        int codepointSize = 0;
        int codepoint = GetCodepointNext(ptr, &codepointSize);
        if (codepointSize <= 0) break;

        if (codepoint == '\n') {
            lines++;
            lineWidth = 0.0f;
            lastSpaceWidth = -1.0f;
            ptr += codepointSize;
            continue;
        }

        float charWidth = MeasureCodepointWidth(font, codepoint, (float)fontSize, 1.0f);
        if (lineWidth + charWidth > (float)maxWidth && lineWidth > 0.0f) {
            lines++;
            if (lastSpaceWidth >= 0.0f) {
                lineWidth = (lineWidth - lastSpaceWidth) + charWidth;
            } else {
                lineWidth = charWidth;
            }
            lastSpaceWidth = -1.0f;
        } else {
            lineWidth += charWidth;
        }

        if (codepoint == ' ') lastSpaceWidth = lineWidth;

        ptr += codepointSize;
    }
    
    *outHeight = lines * LINE_HEIGHT;
    return lines;
}

void DrawTextWrapped(Font font, const char *text, int x, int y, int fontSize, int maxWidth, Color color) {
    if (text == NULL || text[0] == '\0') return;
    
    char line[MAX_MESSAGE_LENGTH] = {0};
    int lineIndex = 0;
    int currentY = y;
    float lineWidth = 0.0f;
    int lastSpaceIndex = -1;

    const char *ptr = text;
    while (*ptr != '\0') {
        int codepointSize = 0;
        int codepoint = GetCodepointNext(ptr, &codepointSize);
        if (codepointSize <= 0) break;

        if (codepoint == '\n') {
            line[lineIndex] = '\0';
            DrawTextEx(font, line, (Vector2){(float)x, (float)currentY}, (float)fontSize, 1.0f, color);
            currentY += LINE_HEIGHT;
            lineIndex = 0;
            lineWidth = 0.0f;
            lastSpaceIndex = -1;
            line[0] = '\0';
            ptr += codepointSize;
            continue;
        }

        float charWidth = MeasureCodepointWidth(font, codepoint, (float)fontSize, 1.0f);
        if (lineWidth + charWidth > (float)maxWidth && lineIndex > 0) {
            if (lastSpaceIndex >= 0) {
                line[lastSpaceIndex] = '\0';
                DrawTextEx(font, line, (Vector2){(float)x, (float)currentY}, (float)fontSize, 1.0f, color);
                currentY += LINE_HEIGHT;

                int overflowStart = lastSpaceIndex + 1;
                int overflowLen = lineIndex - overflowStart;
                memmove(line, line + overflowStart, overflowLen);
                lineIndex = overflowLen;
                line[lineIndex] = '\0';

                lastSpaceIndex = -1;
                for (int j = 0; j < lineIndex; j++) {
                    if (line[j] == ' ') lastSpaceIndex = j;
                }
            } else {
                line[lineIndex] = '\0';
                DrawTextEx(font, line, (Vector2){(float)x, (float)currentY}, (float)fontSize, 1.0f, color);
                currentY += LINE_HEIGHT;
                lineIndex = 0;
                lastSpaceIndex = -1;
                line[0] = '\0';
            }

            lineWidth = (lineIndex > 0) ? MeasureTextEx(font, line, (float)fontSize, 1.0f).x : 0.0f;
        }

        AppendCodepointUTF8(line, &lineIndex, MAX_MESSAGE_LENGTH - 1, codepoint);
        lineWidth = MeasureTextEx(font, line, (float)fontSize, 1.0f).x;
        if (codepoint == ' ') lastSpaceIndex = lineIndex - 1;

        ptr += codepointSize;
    }
    
    if (lineIndex > 0) {
        DrawTextEx(font, line, (Vector2){(float)x, (float)currentY}, (float)fontSize, 1.0f, color);
    }
}

int GetMessageHeight(Font font, const char *text, int maxWidth) {
    if (text == NULL || text[0] == '\0') return LINE_HEIGHT * 2;

    int height = 0;
    MeasureTextWrapped(font, text, FONT_SIZE, maxWidth, &height);
    return height + LINE_HEIGHT;
}

float GetTotalContentHeight(Font font, ChatUI *ui, int maxWidth) {
    float total = 10;  // Initial padding
    for (int i = 0; i < ui->message_count; i++) {
        total += GetMessageHeight(font, ui->messages[i].text, maxWidth) + LINE_HEIGHT / 2;
    }
    return total;
}

// Draw color legend for float values
void DrawFloatColorLegend(int x, int y, int width, int height, float minVal, float maxVal, int diverging) {
    for (int i = 0; i < width; i++) {
        float t = (float)i / (float)width;
        Color c;
        if (diverging) {
            float val = minVal + t * (maxVal - minVal);
            c = FloatToColorDiverging(val, minVal, maxVal);
        } else {
            c = FloatToColorInferno(t);
        }
        DrawLine(x + i, y, x + i, y + height, c);
    }
    
    DrawRectangleLines(x, y, width, height, PLACEHOLDER_COLOR);
    
    char minStr[32], maxStr[32];
    sprintf(minStr, "%.2f", minVal);
    sprintf(maxStr, "%.2f", maxVal);
    DrawTextUI(minStr, x, y + height + UI_SCALE_I(3), 12, PLACEHOLDER_COLOR);
    int maxWidth = MeasureTextUI(maxStr, 12);
    DrawTextUI(maxStr, x + width - maxWidth, y + height + UI_SCALE_I(3), 12, PLACEHOLDER_COLOR);
}

// ----------------------------------------------------------------------------
// Main

int main(int argc, char *argv[]) {
    // Parse command line arguments
    char *checkpoint_path = "Qwen3-0.6B.bin";
    char *system_prompt = NULL;
    float temperature = 0.7f;
    float topp = 0.9f;
    int enable_thinking = 0;
    int ctx_length = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            topp = atof(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            enable_thinking = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            ctx_length = atoi(argv[++i]);
        }
    }
    
    // Initialize raylib
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Qwen3 Visualizer - Activations");
    SetTargetFPS(60);
    
    // Draw loading screen
    BeginDrawing();
    ClearBackground(BG_COLOR);
    const char *loadingText = "Loading Qwen3 model...";
    int textWidth = MeasureTextUI(loadingText, 28);
    DrawTextUI(loadingText, (SCREEN_WIDTH - textWidth) / 2, SCREEN_HEIGHT / 2, 28, TEXT_COLOR);
    EndDrawing();
    
    // Resolve default model path relative to executable so it works from any CWD
    char model_path_buf[1024];
    if (strcmp(checkpoint_path, "Qwen3-0.6B.bin") == 0) {
        const char *exe_dir = GetApplicationDirectory();
        snprintf(model_path_buf, sizeof(model_path_buf), "%sQwen3-0.6B.bin", exe_dir);
        checkpoint_path = model_path_buf;
    }
    
    // Initialize chat state
    ChatState chat_state;
    if (chat_init(&chat_state, checkpoint_path, system_prompt, temperature, topp, 0, enable_thinking, ctx_length) != 0) {
        CloseWindow();
        return 1;
    }
    
    Config *cfg = &chat_state.transformer.config;
    RunState *state = &chat_state.transformer.state;
    
    // Initialize UI state
    ChatUI ui = {0};
    Font chat_font = LoadChatFont(FONT_SIZE);
    int unload_chat_font = (chat_font.texture.id != GetFontDefault().texture.id);
    g_ui_font = chat_font;
    
    // Visualization state
    VisType current_vis = VIS_NETWORK_FULL;  // Start with full network view
    const char *vis_names[] = {
        "Attention Scores",
        "Hidden State (x)",
        "Query Vectors",
        "Top Logits",
        "FFN Activation",
        "Network Graph"
    };
    const char *vis_descs[] = {
        "Attention pattern: rows=heads, cols=positions",
        "Current hidden state reshaped to 2D",
        "Query vectors: rows=heads, cols=head_dim",
        "Top predicted tokens probability",
        "FFN intermediate activation",
        "Full transformer layer visualization"
    };
    
    ActivationVis vis = {0};
    vis.scale = 1.0f;
    vis.offset = (Vector2){0, 0};
    
    // Full network visualization state
    NetworkVis network_vis = {0};
    InitNetworkVis(&network_vis, cfg);
    
    int has_activations = 0;  // Set to 1 after first forward pass
    
    // Main loop
    while (!WindowShouldClose()) {
        int screenWidth = GetScreenWidth();
        int screenHeight = GetScreenHeight();
        
        int chatPanelWidth = CHAT_PANEL_WIDTH;
        int visPanelX = chatPanelWidth + PANEL_MARGIN * 2;
        int visPanelWidth = screenWidth - visPanelX - PANEL_MARGIN;
        int panelHeight = screenHeight - PANEL_MARGIN * 2;
        int chatAreaHeight = panelHeight - INPUT_BOX_HEIGHT - UI_SCALE_I(40);
        int messageMaxWidth = chatPanelWidth - 40;
        
        // Handle visualization type switching
        if (IsKeyPressed(KEY_TAB)) {
            current_vis = (current_vis + 1) % VIS_COUNT;
        }
        if (IsKeyPressed(KEY_ONE)) current_vis = VIS_ATTENTION;
        if (IsKeyPressed(KEY_TWO)) current_vis = VIS_HIDDEN_STATE;
        if (IsKeyPressed(KEY_THREE)) current_vis = VIS_QUERY;
        if (IsKeyPressed(KEY_FOUR)) current_vis = VIS_LOGITS;
        if (IsKeyPressed(KEY_FIVE)) current_vis = VIS_FFN_ACTIVATION;
        if (IsKeyPressed(KEY_SIX) || IsKeyPressed(KEY_G)) current_vis = VIS_NETWORK_FULL;

        // Toggle 3D mode with M key (only for NETWORK_FULL view)
        if (IsKeyPressed(KEY_M) && current_vis == VIS_NETWORK_FULL) {
            network_vis.use_3d_mode = !network_vis.use_3d_mode;
        }
        
        // Toggle texture/voxel rendering with T key (3D mode only)
        if (IsKeyPressed(KEY_T) && current_vis == VIS_NETWORK_FULL && network_vis.use_3d_mode) {
            network_vis.use_textures = !network_vis.use_textures;
        }

        // Update 3D camera (only when in 3D mode and NETWORK_FULL view)
        if (network_vis.use_3d_mode && current_vis == VIS_NETWORK_FULL) {
            float dt = GetFrameTime();
            UpdateFreeFlightCamera(&network_vis.camera, dt);
        }

        // Handle zoom with mouse wheel when hovering over visualization
        Vector2 mouse = GetMousePosition();
        
        // Network visualization area for input handling
        int visAreaY = PANEL_MARGIN + UI_SCALE_I(35);
        int visAreaHeight = panelHeight - UI_SCALE_I(90);
        Rectangle netVisArea = {visPanelX, visAreaY + UI_SCALE_I(40), visPanelWidth, visAreaHeight - UI_SCALE_I(40)};
        
        // Only handle traditional zoom/pan when not in network mode
        if (current_vis != VIS_NETWORK_FULL) {
            if (mouse.x > visPanelX && mouse.x < screenWidth - PANEL_MARGIN) {
                float wheel = GetMouseWheelMove();
                if (wheel != 0) {
                    float oldScale = vis.scale;
                    vis.scale *= (1.0f + wheel * 0.15f);
                    if (vis.scale < 0.5f) vis.scale = 0.5f;
                    if (vis.scale > 20.0f) vis.scale = 20.0f;
                    
                    float zoomFactor = vis.scale / oldScale;
                    vis.offset.x = mouse.x - (mouse.x - vis.offset.x) * zoomFactor;
                    vis.offset.y = mouse.y - (mouse.y - vis.offset.y) * zoomFactor;
                }
            }
            
            // Handle pan with right mouse button
            if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT) && mouse.x > visPanelX) {
                Vector2 delta = GetMouseDelta();
                vis.offset.x += delta.x;
                vis.offset.y += delta.y;
            }
        }
        
        // Reset view with R key (works for both 2D/3D modes)
        if (IsKeyPressed(KEY_R)) {
            vis.scale = 1.0f;
            vis.offset = (Vector2){0, 0};

            // Reset 3D camera to initial position
            if (network_vis.use_3d_mode && current_vis == VIS_NETWORK_FULL) {
                Vector3 start_pos = {25.0f, 20.0f, -20.0f};
                Vector3 look_at = {0.0f, 0.0f, 24.0f};
                InitFreeFlightCamera(&network_vis.camera, start_pos, look_at);
            }
        }
        
        // Handle chat input (UTF-8, IME-friendly)
        if (!ui.is_generating) {
            int textInputThisFrame = 0;
            int key = GetCharPressed();
            while (key > 0) {
                if (key >= 32 && key != 127) {
                    AppendCodepointUTF8(ui.input_buffer, &ui.input_length, MAX_MESSAGE_LENGTH - 1, key);
                    textInputThisFrame = 1;
                }
                key = GetCharPressed();
            }
            
            if (IsKeyPressed(KEY_BACKSPACE) || IsKeyPressedRepeat(KEY_BACKSPACE)) {
                if (ui.input_length > 0) {
                    int offset = GetPreviousCodepointOffset(ui.input_buffer, ui.input_length);
                    ui.input_length -= offset;
                    ui.input_buffer[ui.input_length] = '\0';
                }
            }
            
            if (IsKeyPressed(KEY_ENTER) && !textInputThisFrame && ui.input_length > 0) {
                if (ui.message_count < MAX_MESSAGES) {
                    strncpy(ui.messages[ui.message_count].text, ui.input_buffer, MAX_MESSAGE_LENGTH - 1);
                    ui.messages[ui.message_count].is_user = 1;
                    ui.messages[ui.message_count].is_complete = 1;
                    ui.message_count++;
                }
                
                chat_submit_prompt(&chat_state, ui.input_buffer);
                ui.is_generating = 1;
                
                if (ui.message_count < MAX_MESSAGES) {
                    ui.messages[ui.message_count].text[0] = '\0';
                    ui.messages[ui.message_count].is_user = 0;
                    ui.messages[ui.message_count].is_complete = 0;
                    ui.message_count++;
                }
                
                ui.input_buffer[0] = '\0';
                ui.input_length = 0;
                
                ui.target_scroll = GetTotalContentHeight(chat_font, &ui, messageMaxWidth) - chatAreaHeight + UI_SCALE_I(50);
                if (ui.target_scroll < 0) ui.target_scroll = 0;
            }
        }
        
        // Generate tokens and update visualization
        if (ui.is_generating) {
            const char *token = chat_generate_next(&chat_state);
            
            // Update visualization with current activations
            if (chat_state.pos > 0) {
                has_activations = 1;
                
                // Update network vis data for 2D mode (3D mode updates in its own function)
                if (current_vis != VIS_NETWORK_FULL || !network_vis.use_3d_mode) {
                    UpdateNetworkVisData(&network_vis, state, &chat_state.transformer.weights, cfg, chat_state.pos - 1);
                }
                
                switch (current_vis) {
                    case VIS_ATTENTION:
                        UpdateAttentionVis(&vis, state, cfg, chat_state.pos - 1);
                        break;
                    case VIS_HIDDEN_STATE:
                        UpdateHiddenStateVis(&vis, state->x, cfg->dim);
                        break;
                    case VIS_QUERY:
                        UpdateQueryVis(&vis, state->q, cfg->n_heads, cfg->head_dim);
                        break;
                    case VIS_LOGITS:
                        UpdateLogitsVis(&vis, state->logits, cfg->vocab_size, 50);
                        break;
                    case VIS_FFN_ACTIVATION:
                        UpdateHiddenStateVis(&vis, state->hb, cfg->hidden_dim);
                        break;
                    case VIS_NETWORK_FULL:
                        // Data already updated above
                        break;
                    default:
                        break;
                }
            }
            
            if (token != NULL && token[0] != '\0') {
                int msg_idx = ui.message_count - 1;
                int current_len = strlen(ui.messages[msg_idx].text);
                int token_len = strlen(token);
                if (current_len + token_len < MAX_MESSAGE_LENGTH - 1) {
                    strcat(ui.messages[msg_idx].text, token);
                }
                
                float contentHeight = GetTotalContentHeight(chat_font, &ui, messageMaxWidth);
                if (contentHeight > chatAreaHeight) {
                    ui.target_scroll = contentHeight - chatAreaHeight + UI_SCALE_I(20);
                }
            }
            
            if (chat_is_done(&chat_state)) {
                ui.is_generating = 0;
                if (ui.message_count > 0) {
                    ui.messages[ui.message_count - 1].is_complete = 1;
                }
            }
        }
        
        // Handle chat scroll
        if (mouse.x < chatPanelWidth + PANEL_MARGIN) {
            float wheel = GetMouseWheelMove();
            if (wheel != 0) {
                ui.target_scroll -= wheel * UI_SCALE_I(30);
                float maxScroll = GetTotalContentHeight(chat_font, &ui, messageMaxWidth) - chatAreaHeight + UI_SCALE_I(20);
                if (ui.target_scroll < 0) ui.target_scroll = 0;
                if (ui.target_scroll > maxScroll && maxScroll > 0) ui.target_scroll = maxScroll;
            }
        }
        
        ui.scroll_offset += (ui.target_scroll - ui.scroll_offset) * 0.15f;

        int inputY = PANEL_MARGIN + panelHeight - INPUT_BOX_HEIGHT - UI_SCALE_I(5);
        if (!ui.is_generating) {
            float textWidth = MeasureTextEx(chat_font, ui.input_buffer, (float)FONT_SIZE, 1.0f).x;
            float caretX = (float)(PANEL_MARGIN + 15) + textWidth;
            float caretY = (float)(inputY + (INPUT_BOX_HEIGHT - FONT_SIZE) / 2);
            Vector2 dpi = GetWindowScaleDPI();
            int imeX = (int)(caretX * dpi.x + 0.5f);
            int imeY = (int)((caretY + FONT_SIZE) * dpi.y + 0.5f);
            UpdateImeWindowPosition(imeX, imeY);
        }
        
        // Drawing
        BeginDrawing();
        ClearBackground(BG_COLOR);
        
        // ===== LEFT PANEL: Chat =====
        DrawRectangle(PANEL_MARGIN, PANEL_MARGIN, chatPanelWidth, panelHeight, CHAT_BG_COLOR);
        
        DrawRectangle(PANEL_MARGIN, PANEL_MARGIN, chatPanelWidth, UI_SCALE_I(30), TITLE_BG_COLOR);
        DrawTextUI("Chat", PANEL_MARGIN + UI_SCALE_I(10), PANEL_MARGIN + UI_SCALE_I(6), 18, TEXT_COLOR);

        char statusText[64];
        sprintf(statusText, "%d/%d", chat_state.pos, cfg->seq_len);
        int statusWidth = MeasureTextUI(statusText, 16);
        DrawTextUI(statusText, PANEL_MARGIN + chatPanelWidth - statusWidth - UI_SCALE_I(10), PANEL_MARGIN + UI_SCALE_I(8), 16, PLACEHOLDER_COLOR);
        
        int chatY = PANEL_MARGIN + UI_SCALE_I(35);
        BeginScissorMode(PANEL_MARGIN, chatY, chatPanelWidth, chatAreaHeight);
        
        float y = chatY + UI_SCALE_I(10) - ui.scroll_offset;
        for (int i = 0; i < ui.message_count; i++) {
            ChatMessage *msg = &ui.messages[i];
            int msgX = PANEL_MARGIN + 15;
            int msgW = messageMaxWidth - 30;

            // Calculate text height for this message
            int textHeight;
            MeasureTextWrapped(chat_font, msg->text, FONT_SIZE, msgW, &textHeight);
            int msgHeight = textHeight + LINE_HEIGHT;

            if (y + msgHeight > chatY && y < chatY + chatAreaHeight) {
                // Terminal style: prefix with ">" for user, no prefix for assistant
                Color prefixColor = msg->is_user ? ACCENT_COLOR : (Color){100, 180, 100, 255};
                const char *prefix = msg->is_user ? "> " : "";

                if (prefix[0] != '\0') {
                    DrawTextEx(chat_font, prefix, (Vector2){(float)msgX, y}, (float)FONT_SIZE, 1.0f, prefixColor);
                }

                int textX = msgX + (msg->is_user ? (int)MeasureTextEx(chat_font, prefix, (float)FONT_SIZE, 1.0f).x : 0);
                DrawTextWrapped(chat_font, msg->text, textX, y,
                               FONT_SIZE, msgW - (textX - msgX), TEXT_COLOR);

                if (!msg->is_complete && !msg->is_user && (int)(GetTime() * 3) % 2) {
                    DrawTextEx(chat_font, "_", (Vector2){(float)(textX + MeasureTextEx(chat_font, msg->text, (float)FONT_SIZE, 1.0f).x), y}, (float)FONT_SIZE, 1.0f, ACCENT_COLOR);
                }
            }

            y += msgHeight + LINE_HEIGHT / 2;
        }
        
        EndScissorMode();
        
        DrawRectangleRounded((Rectangle){PANEL_MARGIN + 5, inputY, chatPanelWidth - 10, INPUT_BOX_HEIGHT}, 0.1f, 8, INPUT_BG_COLOR);
        
        if (ui.input_length > 0) {
            DrawTextEx(chat_font, ui.input_buffer, (Vector2){(float)(PANEL_MARGIN + 15), (float)(inputY + (INPUT_BOX_HEIGHT - FONT_SIZE) / 2)}, (float)FONT_SIZE, 1.0f, TEXT_COLOR);
            if ((int)(GetTime() * 2) % 2) {
                int cursorX = (int)(PANEL_MARGIN + 15 + MeasureTextEx(chat_font, ui.input_buffer, (float)FONT_SIZE, 1.0f).x);
                DrawTextEx(chat_font, "|", (Vector2){(float)cursorX, (float)(inputY + (INPUT_BOX_HEIGHT - FONT_SIZE) / 2)}, (float)FONT_SIZE, 1.0f, ACCENT_COLOR);
            }
        } else {
            const char *placeholder = ui.is_generating ? "Generating..." : "Type message...";
            DrawTextUI(placeholder, PANEL_MARGIN + 15, inputY + (INPUT_BOX_HEIGHT - FONT_SIZE) / 2, BASE_FONT_SIZE, PLACEHOLDER_COLOR);
        }
        
        // ===== RIGHT PANEL: Activation Visualization =====
        DrawRectangle(visPanelX, PANEL_MARGIN, visPanelWidth, panelHeight, VIS_BG_COLOR);
        
        // Title bar
        DrawRectangle(visPanelX, PANEL_MARGIN, visPanelWidth, UI_SCALE_I(30), TITLE_BG_COLOR);
        char visTitle[128];
        if (current_vis == VIS_NETWORK_FULL) {
            sprintf(visTitle, "%s  Layer %d/%d", vis_names[current_vis],
                    network_vis.view.current_layer + 1, cfg->n_layers);
        } else {
            sprintf(visTitle, "%s  [%dx%d]", vis_names[current_vis], vis.width, vis.height);
        }
        DrawTextUI(visTitle, visPanelX + UI_SCALE_I(10), PANEL_MARGIN + UI_SCALE_I(6), 18, TEXT_COLOR);

        // Live indicator
        if (ui.is_generating) {
            DrawCircle(visPanelX + visPanelWidth - UI_SCALE_I(20), PANEL_MARGIN + UI_SCALE_I(15), UI_SCALE_I(7), (Color){255, 80, 80, 255});
            DrawTextUI("LIVE", visPanelX + visPanelWidth - UI_SCALE_I(60), PANEL_MARGIN + UI_SCALE_I(8), 14, (Color){255, 80, 80, 255});
        }
        
        // Visualization area (redefine since we may not have set it earlier)
        int visAreaY_draw = PANEL_MARGIN + UI_SCALE_I(35);
        int visAreaHeight_draw = panelHeight - UI_SCALE_I(90);
        
        if (current_vis == VIS_NETWORK_FULL) {
            // Full network visualization
            Rectangle netArea = {visPanelX, visAreaY_draw, visPanelWidth, visAreaHeight_draw};

            // Handle network vis input (only in 2D mode, 3D has its own controls)
            if (!network_vis.use_3d_mode) {
                HandleNetworkVisInput(&network_vis, cfg->n_layers, netArea, mouse);
            }

            // Draw layer navigation
            DrawLayerNav(&network_vis, netArea, cfg->n_layers);

            // Adjust area for content (below nav bar)
            Rectangle contentArea = {visPanelX, visAreaY_draw + UI_SCALE_I(40), visPanelWidth, visAreaHeight_draw - UI_SCALE_I(40)};

            BeginScissorMode((int)contentArea.x, (int)contentArea.y, (int)contentArea.width, (int)contentArea.height);

            if (has_activations) {
                if (network_vis.use_3d_mode) {
                    // === 3D RENDERING MODE ===
                    
                    // Update 3D data and ray picking (only on mouse move)
                    UpdateNetworkVis3D(&network_vis, state, &chat_state.transformer.weights, cfg, chat_state.pos - 1, mouse);
                    HandleNetworkVisInput3D(&network_vis, cfg->n_layers);
                    
                    BeginMode3D(network_vis.camera.camera);
                    {
                        // Render full 3D network with voxels
                        DrawNetworkVis3D(&network_vis, cfg);
                    }
                    EndMode3D();

                    // 2D overlay: Mode indicator and controls
                    DrawRectangle(visPanelX + UI_SCALE_I(5), visAreaY_draw + UI_SCALE_I(45), UI_SCALE_I(350), UI_SCALE_I(70), (Color){0, 0, 0, 180});
                    const char* modeStr = network_vis.use_textures ? "TEXTURE" : "VOXEL";
                    char modeText[64];
                    sprintf(modeText, "3D MODE [%s] - [M]2D/3D [T]Tex/Vox", modeStr);
                    DrawTextUI(modeText, visPanelX + UI_SCALE_I(10), visAreaY_draw + UI_SCALE_I(50), 14, (Color){100, 255, 100, 255});
                    DrawTextUI("[WASD] Move  [Space/Ctrl] Up/Down  [RClick] Look", visPanelX + UI_SCALE_I(10), visAreaY_draw + UI_SCALE_I(68), 12, WHITE);
                    DrawTextUI("[</>] Layer  [W] Toggle Weights  [R] Reset Cam", visPanelX + UI_SCALE_I(10), visAreaY_draw + UI_SCALE_I(84), 12, WHITE);
                    
                    // Show picked node info
                    if (network_vis.has_pick && network_vis.picked_node >= 0) {
                        VisNode* node = &network_vis.nodes[network_vis.picked_node];
                        char info[128];
                        if (network_vis.hover_row >= 0 && network_vis.hover_col >= 0) {
                            sprintf(info, "%s [%d,%d] = %.4f", node->name, 
                                   network_vis.hover_row, network_vis.hover_col, network_vis.hover_value);
                        } else {
                            sprintf(info, "%s (%s)", node->name, node->label);
                        }
                        DrawRectangle(visPanelX + UI_SCALE_I(5), visAreaY_draw + UI_SCALE_I(120), UI_SCALE_I(280), UI_SCALE_I(28), (Color){20, 20, 28, 240});
                        DrawRectangleLines(visPanelX + UI_SCALE_I(5), visAreaY_draw + UI_SCALE_I(120), UI_SCALE_I(280), UI_SCALE_I(28), YELLOW);
                        DrawTextUI(info, visPanelX + UI_SCALE_I(12), visAreaY_draw + UI_SCALE_I(126), 14, YELLOW);
                    }
                } else {
                    // === 2D RENDERING MODE ===
                    DrawNetworkVis(&network_vis, contentArea, mouse, cfg);
                }
            } else {
                const char *waitText = "Waiting for activations...";
                const char *hintText = "Type a message to see the model's internal states";
                int tw = MeasureTextUI(waitText, 22);
                int hw = MeasureTextUI(hintText, 16);
                DrawTextUI(waitText, visPanelX + (visPanelWidth - tw) / 2, visAreaY_draw + visAreaHeight_draw / 2 - UI_SCALE_I(20), 22, PLACEHOLDER_COLOR);
                DrawTextUI(hintText, visPanelX + (visPanelWidth - hw) / 2, visAreaY_draw + visAreaHeight_draw / 2 + UI_SCALE_I(10), 16, (Color){80, 80, 90, 255});

                // Hint about 3D mode
                const char *mode3d = "Press [M] to toggle 3D mode when activated";
                int mw = MeasureTextUI(mode3d, 12);
                DrawTextUI(mode3d, visPanelX + (visPanelWidth - mw) / 2, visAreaY_draw + visAreaHeight_draw / 2 + UI_SCALE_I(40), 12, (Color){100, 100, 110, 255});
            }

            EndScissorMode();

            // Draw hover tooltip (2D mode only)
            if (has_activations && !network_vis.use_3d_mode) {
                DrawHoverTooltip(&network_vis, mouse);
            }
        } else {
            // Traditional heatmap visualization
            BeginScissorMode(visPanelX, visAreaY_draw, visPanelWidth, visAreaHeight_draw);
            
            if (has_activations && vis.valid) {
                float displayWidth = vis.width * vis.scale;
                float displayHeight = vis.height * vis.scale;
                float centerX = visPanelX + visPanelWidth / 2 + vis.offset.x;
                float centerY = visAreaY_draw + visAreaHeight_draw / 2 + vis.offset.y;
                
                Rectangle srcRect = {0, 0, vis.width, vis.height};
                Rectangle destRect = {centerX - displayWidth / 2, centerY - displayHeight / 2, displayWidth, displayHeight};
                
                DrawTexturePro(vis.texture, srcRect, destRect, (Vector2){0, 0}, 0, WHITE);
                
                // Grid lines for large zoom
                if (vis.scale > 4.0f) {
                    Color gridColor = (Color){255, 255, 255, 30};
                    for (int gx = 0; gx <= vis.width; gx++) {
                        float x = destRect.x + gx * vis.scale;
                        DrawLine(x, destRect.y, x, destRect.y + destRect.height, gridColor);
                    }
                    for (int gy = 0; gy <= vis.height; gy++) {
                        float yy = destRect.y + gy * vis.scale;
                        DrawLine(destRect.x, yy, destRect.x + destRect.width, yy, gridColor);
                    }
                }
            } else {
                const char *waitText = "Waiting for activations...";
                const char *hintText = "Type a message to see the model's internal states";
                int tw = MeasureTextUI(waitText, 22);
                int hw = MeasureTextUI(hintText, 16);
                DrawTextUI(waitText, visPanelX + (visPanelWidth - tw) / 2, visAreaY_draw + visAreaHeight_draw / 2 - UI_SCALE_I(20), 22, PLACEHOLDER_COLOR);
                DrawTextUI(hintText, visPanelX + (visPanelWidth - hw) / 2, visAreaY_draw + visAreaHeight_draw / 2 + UI_SCALE_I(10), 16, (Color){80, 80, 90, 255});
            }
            
            EndScissorMode();
            
            // Color legend
            if (has_activations && vis.valid) {
                int isDiverging = (current_vis != VIS_ATTENTION && current_vis != VIS_LOGITS);
                DrawFloatColorLegend(visPanelX + visPanelWidth - UI_SCALE_I(170), visAreaY_draw + UI_SCALE_I(10), UI_SCALE_I(150), UI_SCALE_I(12), vis.min_val, vis.max_val, isDiverging);
            }
        }
        
        // Info bar
        int infoY = PANEL_MARGIN + panelHeight - UI_SCALE_I(50);
        DrawRectangle(visPanelX, infoY, visPanelWidth, UI_SCALE_I(40), TITLE_BG_COLOR);

        DrawTextUI(vis_descs[current_vis], visPanelX + UI_SCALE_I(10), infoY + UI_SCALE_I(5), 14, TEXT_COLOR);

        char controlsText[196];
        if (current_vis == VIS_NETWORK_FULL) {
            if (network_vis.use_3d_mode) {
                sprintf(controlsText, "[M] 2D/3D  [WASD] Move  [RClick] Look  [R] Reset  [<-/->] Layer");
            } else {
                sprintf(controlsText, "[M] 2D/3D  [<-/->] Layer  [W] Weights  [+/-] Zoom  [R] Reset");
            }
        } else {
            sprintf(controlsText, "[1-6/G] View  [Tab] Next  [R] Reset  [Scroll] Zoom  [Right-drag] Pan");
        }
        DrawTextUI(controlsText, visPanelX + UI_SCALE_I(10), infoY + UI_SCALE_I(22), 12, PLACEHOLDER_COLOR);

        char zoomText[32];
        if (current_vis == VIS_NETWORK_FULL) {
            sprintf(zoomText, "%.1fx", network_vis.view.zoom);
        } else {
            sprintf(zoomText, "%.1fx", vis.scale);
        }
        int zw = MeasureTextUI(zoomText, 16);
        DrawTextUI(zoomText, visPanelX + visPanelWidth - zw - UI_SCALE_I(10), infoY + UI_SCALE_I(12), 16, ACCENT_COLOR);
        
        EndDrawing();
    }
    
    // Cleanup
    FreeActivationVis(&vis);
    
    // Free 3D voxel caches
    for (int i = 0; i < VNODE_COUNT; i++) {
        FreeVoxelCache(&network_vis.voxel_caches[i]);
    }
    
    // Free texture caches
    for (int i = 0; i < VNODE_COUNT; i++) {
        FreeNodeTextureCache(&network_vis.texture_caches[i]);
    }
    
    if (unload_chat_font) UnloadFont(chat_font);

    chat_free(&chat_state);
    CloseWindow();
    
    return 0;
}


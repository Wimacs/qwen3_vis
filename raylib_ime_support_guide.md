# Raylib IME 支持实现指南

## 目标

为 raylib 添加中文/日文等 IME（输入法）支持，使输入法候选框能够正确显示在文字输入位置旁边，类似 Chrome 浏览器的行为；并确保 **中文字体能正确加载并渲染**（避免显示 `?`）。

## 技术方案（基于实战结果）

**只保留 SDL 方案**，保证跨平台（Windows/macOS/Linux）一致，并避免平台特化代码。

### 方案：SDL 后端（跨平台、工程改动在 raylib 平台层）
- 优点：跨平台、候选框位置可控（SDL2: `SDL_SetTextInputRect` / SDL3: `SDL_SetTextInputArea`）
- 代价：需要改 raylib 平台层，并切换到 SDL 后端构建

## 前置要求

- raylib 源码（5.0 或更高版本）
- （方案 A）SDL2/SDL3 开发库
- 支持中文的字体文件（建议：`NotoSansCJK-Regular.ttc` 或 `NotoSansCJK-Regular.otf`）

## 实现步骤

> **重要经验**：中文显示成 `?` 的常见原因不是 IME，而是字体没真正加载成功，或 `TTC` 字体集合未被解析。

### 第一步：切换 raylib 到 SDL 后端（必做）

#### 1.1 在你的 CMake 中指定 SDL 平台

```cmake
# 在 add_subdirectory(3rd/raylib) 之前：
set(PLATFORM "SDL" CACHE STRING "" FORCE)
```

#### 1.2 引入 SDL（建议使用子目录）

```cmake
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(3rd/SDL2 EXCLUDE_FROM_ALL)
```

> 实测你拉的是 SDL3 代码仓库，但目录叫 SDL2；raylib 会自动识别 SDL3 target。

---

### 第二步：修改 SDL 后端源文件（IME 支持）

文件路径：`src/platforms/rcore_desktop_sdl.c`

#### 2.1 在文件顶部添加 IME 相关变量

在 `PlatformData` 结构体或全局变量区域添加：

```c
// IME Support
static bool imeEnabled = false;
static SDL_Rect imeRect = { 0, 0, 1, 20 };
static IMECompositionInfo imeComposition = { 0 };
```

#### 2.2 修改 InitPlatform() 函数（重要：Hint 必须在 SDL_Init 之前）

**必须在 `SDL_Init()` 之前设置**，否则 SDL3 不会发预编辑事件。

```c
#if defined(USING_VERSION_SDL3)
SDL_SetHint(SDL_HINT_IME_IMPLEMENTED_UI, "composition");
#else
SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif
```

> 真实踩坑：放在 `SDL_Init()` 之后会导致 **预编辑事件完全不触发**。

#### 2.3 添加 IME 控制函数（SDL2 vs SDL3 差异）

在文件末尾（`// Module Functions Definition` 区域）添加以下函数：

```c
//----------------------------------------------------------------------------------
// Module Functions Definition: IME Support
//----------------------------------------------------------------------------------

// Start text input mode (activate IME)
void StartTextInput(void)
{
#if defined(USING_VERSION_SDL3)
    SDL_StartTextInput(platform.window);
#else
    SDL_StartTextInput();
#endif
    imeEnabled = true;
}

// Stop text input mode (deactivate IME)
void StopTextInput(void)
{
#if defined(USING_VERSION_SDL3)
    SDL_StopTextInput(platform.window);
#else
    SDL_StopTextInput();
#endif
    imeEnabled = false;
}

// Check if text input is active
bool IsTextInputActive(void)
{
#if defined(USING_VERSION_SDL3)
    return SDL_TextInputActive(platform.window);
#else
    return SDL_IsTextInputActive();
#endif
}

// Set IME candidate window position
// Call this every frame with the current text cursor position
void SetTextInputRect(int x, int y, int width, int height)
{
    imeRect.x = x;
    imeRect.y = y;
    imeRect.w = width;
    imeRect.h = height;
#if defined(USING_VERSION_SDL3)
    SDL_SetTextInputArea(platform.window, &imeRect, 0);
#else
    SDL_SetTextInputRect(&imeRect);
#endif
}

// Get current IME rect
Rectangle GetTextInputRect(void)
{
    return (Rectangle){ (float)imeRect.x, (float)imeRect.y, 
                        (float)imeRect.w, (float)imeRect.h };
}

// Get IME preedit info
IMECompositionInfo GetIMECompositionInfo(void)
{
    return imeComposition;
}
```

#### 2.4 关键点：候选框位置的坐标系（必须处理 DPI/缩放）

`SDL_SetTextInputRect` 需要的是**窗口客户端区域的像素坐标**，而不是 world 坐标或缩放后的虚拟坐标。  
如果你用了高 DPI、摄像机（`BeginMode2D`）、或 RenderTexture 做缩放/信箱渲染，必须把光标位置转换到**屏幕像素**再传入：

```c
// 1) 得到光标在屏幕空间的坐标 (screen-space, 逻辑像素)
Vector2 caretScreen = caretPos; // 默认就是 screen-space
// 如果在 2D 摄像机下绘制 UI，先转到屏幕坐标：
// Vector2 caretScreen = GetWorldToScreen2D(caretWorldPos, camera);

// 2) 处理高 DPI：优先使用渲染尺寸比例（更可靠）
float scaleX = (float)GetRenderWidth() / (float)GetScreenWidth();
float scaleY = (float)GetRenderHeight() / (float)GetScreenHeight();
int imeX = (int)(caretScreen.x * scaleX + 0.5f);
int imeY = (int)((caretScreen.y + lineHeight) * scaleY + 0.5f);
int imeH = (int)(lineHeight * scaleY + 0.5f);

// 3) 设置候选框位置（每帧更新）
SetTextInputRect(imeX, imeY, 1, imeH);
```

> 经验：候选框通常显示在**光标矩形的右下方**，所以把 `y` 设为 `caretY + lineHeight` 更稳。

#### 2.5 处理 SDL_TEXTEDITING 事件（必做：预编辑文本显示）

在 `PollInputEvents()` 函数中找到 `SDL_TEXTINPUT` 事件处理，在其附近添加：

```c
#if defined(USING_VERSION_SDL3)
case SDL_EVENT_TEXT_EDITING:
{
    strncpy(imeComposition.text, event.edit.text, sizeof(imeComposition.text) - 1);
    imeComposition.text[sizeof(imeComposition.text) - 1] = '\0';
    imeComposition.cursor = event.edit.start;
    imeComposition.length = event.edit.length;
} break;
#else
case SDL_TEXTEDITING:
{
    strncpy(imeComposition.text, event.edit.text, sizeof(imeComposition.text) - 1);
    imeComposition.text[sizeof(imeComposition.text) - 1] = '\0';
    imeComposition.cursor = event.edit.start;
    imeComposition.length = event.edit.length;
} break;
#endif
```

---

### 第三步：修改 raylib.h 头文件

文件路径：`src/raylib.h`

在 `// Window-related functions` 区域或文件末尾添加函数声明：

```c
//------------------------------------------------------------------------------------
// IME Support Functions (SDL backend only)
//------------------------------------------------------------------------------------
#if defined(PLATFORM_DESKTOP_SDL)
RLAPI void StartTextInput(void);                                    // Start text input mode (activate IME)
RLAPI void StopTextInput(void);                                     // Stop text input mode (deactivate IME)
RLAPI bool IsTextInputActive(void);                                 // Check if text input is active
RLAPI void SetTextInputRect(int x, int y, int width, int height);   // Set IME candidate window position
RLAPI Rectangle GetTextInputRect(void);                             // Get current IME rect

typedef struct {
    char text[64];      // Preedit text (UTF-8)
    int cursor;         // Cursor position in preedit
    int length;         // Selection length in preedit
} IMECompositionInfo;

RLAPI IMECompositionInfo GetIMECompositionInfo(void);              // Get current IME preedit info
#endif
```

**注意**：如果希望在非 SDL 后端也能编译（只是功能不可用），可以在其他后端文件中添加空实现：

```c
// 在 rcore_desktop_glfw.c 中添加（如果需要）
void StartTextInput(void) { }
void StopTextInput(void) { }
bool IsTextInputActive(void) { return false; }
void SetTextInputRect(int x, int y, int width, int height) { }
Rectangle GetTextInputRect(void) { return (Rectangle){ 0 }; }
```

### 第四步：编译 raylib（使用 SDL 后端）

#### 使用 Makefile

```bash
cd raylib/src
make clean
make PLATFORM=PLATFORM_DESKTOP_SDL
```

确保系统已安装 SDL2 开发库：
- Ubuntu/Debian: `sudo apt install libsdl2-dev`
- macOS: `brew install sdl2`
- Windows: 下载 SDL2-devel 并配置路径

#### 使用 CMake（需要额外配置）

CMake 需要指定 `PLATFORM=SDL` 并确保能找到 SDL2/SDL3：

```cmake
# 在 add_subdirectory(3rd/raylib) 之前：
set(PLATFORM "SDL" CACHE STRING "" FORCE)
add_subdirectory(3rd/raylib)
```

> SDL2/SDL3 需要在系统已安装，或通过 add_subdirectory 引入对应源码。

### 第五步：创建测试示例

创建文件 `examples/text/text_input_ime.c`：

```c
/*******************************************************************************************
*   raylib [text] example - IME text input (Chinese/Japanese support)
********************************************************************************************/

#include "raylib.h"
#include <string.h>
#include <stdlib.h>

#define MAX_INPUT_CHARS 256

// UTF-8 aware backspace
static int GetPreviousCodepointOffset(const char *text, int bytePos)
{
    if (bytePos <= 0) return 0;
    
    int offset = 1;
    while (bytePos - offset > 0 && ((unsigned char)text[bytePos - offset] & 0xC0) == 0x80)
        offset++;
    
    return offset;
}

    // Keep IME candidate window aligned with caret (DPI-aware)
    static void UpdateImeRect(Rectangle inputBox, Font font, const char *text, int fontSize)
    {
        Vector2 textSize = MeasureTextEx(font, text, (float)fontSize, 1);
        Vector2 caret = (Vector2){ inputBox.x + 10 + textSize.x, inputBox.y + 10 };
        float scaleX = (float)GetRenderWidth() / (float)GetScreenWidth();
        float scaleY = (float)GetRenderHeight() / (float)GetScreenHeight();
        int imeX = (int)(caret.x * scaleX + 0.5f);
        int imeY = (int)((caret.y + fontSize) * scaleY + 0.5f);
        int imeH = (int)(fontSize * scaleY + 0.5f);
        SetTextInputRect(imeX, imeY, 1, imeH);
    }

int main(void)
{
    // Initialization
    const int screenWidth = 800;
    const int screenHeight = 450;

    InitWindow(screenWidth, screenHeight, "raylib [text] example - IME input");

    // Load Chinese font (确保真实加载到 CJK 字体)
    // 需要准备支持中文的字体文件
    int codepoints[20000];
    int codepointCount = 0;
    
    // ASCII
    for (int i = 32; i < 127; i++) codepoints[codepointCount++] = i;
    // CJK Unified Ideographs (中文常用字)
    for (int i = 0x4E00; i <= 0x9FFF; i++) codepoints[codepointCount++] = i;
    // CJK Symbols and Punctuation
    for (int i = 0x3000; i <= 0x303F; i++) codepoints[codepointCount++] = i;
    // Hiragana
    for (int i = 0x3040; i <= 0x309F; i++) codepoints[codepointCount++] = i;
    // Katakana
    for (int i = 0x30A0; i <= 0x30FF; i++) codepoints[codepointCount++] = i;
    
    // 尝试加载中文字体（按优先级）
    Font font = { 0 };
    const char *fontPaths[] = {
        // 建议放在项目根目录或 resources 下
        "NotoSansCJK-Regular.ttc",
        "NotoSansCJK-Regular.otf",
        "resources/NotoSansCJK-Regular.ttc",
        "resources/simhei.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        NULL
    };
    
    for (int i = 0; fontPaths[i] != NULL; i++) {
        if (FileExists(fontPaths[i])) {
            font = LoadFontEx(fontPaths[i], 32, codepoints, codepointCount);
            if (font.texture.id != 0) {
                TraceLog(LOG_INFO, "Loaded font: %s", fontPaths[i]);
                break;
            }
        }
    }
    
    if (font.texture.id == 0) {
        TraceLog(LOG_WARNING, "Could not load CJK font, using default");
        font = GetFontDefault();
    }

    char text[MAX_INPUT_CHARS + 1] = "\0";
    int textLength = 0;

    Rectangle inputBox = { 100, 180, 600, 50 };
    bool inputActive = false;

    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        // Update
        Vector2 mousePos = GetMousePosition();
        
        // Check if input box clicked
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON))
        {
            if (CheckCollisionPointRec(mousePos, inputBox))
            {
                inputActive = true;
                StartTextInput();

                // 立刻更新 IME 候选框位置（DPI-aware）
                UpdateImeRect(inputBox, font, text, 28);
            }
            else
            {
                inputActive = false;
                StopTextInput();
            }
        }
        
        // Handle text input
        if (inputActive)
        {
            // 更新 IME 候选框位置（每帧更新以跟随光标）
            UpdateImeRect(inputBox, font, text, 28);
            
            // Get unicode character input
            int key = GetCharPressed();
            while (key > 0)
            {
                // Encode codepoint to UTF-8
                if (textLength < MAX_INPUT_CHARS - 4)  // Reserve space for UTF-8
                {
                    int byteCount = 0;
                    const char *utf8 = CodepointToUTF8(key, &byteCount);
                    
                    for (int i = 0; i < byteCount; i++)
                    {
                        text[textLength++] = utf8[i];
                    }
                    text[textLength] = '\0';
                }
                
                key = GetCharPressed();
            }
            
            // Handle backspace (UTF-8 aware)
            if (IsKeyPressed(KEY_BACKSPACE) || IsKeyPressedRepeat(KEY_BACKSPACE))
            {
                if (textLength > 0)
                {
                    int offset = GetPreviousCodepointOffset(text, textLength);
                    textLength -= offset;
                    text[textLength] = '\0';
                }
            }
        }

        // Draw
        BeginDrawing();
        
            ClearBackground(RAYWHITE);
            
            DrawText("Click on the input box and type (supports Chinese/Japanese IME):", 
                     100, 140, 20, GRAY);
            
            // Draw input box
            DrawRectangleRec(inputBox, inputActive ? LIGHTGRAY : Color{230, 230, 230, 255});
            DrawRectangleLinesEx(inputBox, 2, inputActive ? BLUE : DARKGRAY);
            
            // Draw text
            DrawTextEx(font, text, (Vector2){ inputBox.x + 10, inputBox.y + 10 }, 28, 1, BLACK);
            
            // Draw cursor (blinking)
            if (inputActive && ((int)(GetTime() * 2) % 2 == 0))
            {
                Vector2 textSize = MeasureTextEx(font, text, 28, 1);
                DrawRectangle((int)(inputBox.x + 10 + textSize.x), 
                             (int)(inputBox.y + 10), 2, 30, BLACK);
            }
            
            // Instructions
            DrawText("Press ESC to exit", 100, 300, 20, GRAY);
            DrawText("IME Status: ", 100, 340, 20, GRAY);
            DrawText(IsTextInputActive() ? "ACTIVE" : "INACTIVE", 220, 340, 20, 
                     IsTextInputActive() ? GREEN : RED);
            
            // Show text length info
            DrawText(TextFormat("Text length: %d bytes", textLength), 100, 380, 20, GRAY);

        EndDrawing();
    }

    // Cleanup
    StopTextInput();
    if (font.texture.id != GetFontDefault().texture.id) UnloadFont(font);
    CloseWindow();

    return 0;
}
```

### 第五步：编译并测试示例

```bash
# 确保 raylib 已经用 SDL 后端编译
cd raylib/examples

# 编译示例（需要链接 SDL2）
gcc text/text_input_ime.c -o text_input_ime \
    -I../src -L../src -lraylib \
    $(pkg-config --cflags --libs sdl2) \
    -lm -lpthread -ldl

# 运行
./text_input_ime
```

## 验证清单

- [ ] SDL hint 在 **SDL_Init 之前**调用（SDL3 必须）
- [ ] StartTextInput/StopTextInput 在输入框聚焦/失焦时调用
- [ ] SetTextInputRect 每帧更新候选框位置（输入框聚焦时）
- [ ] 已将光标位置转换到窗口像素坐标（含 DPI/缩放/摄像机）
- [ ] 中文字体正确加载
- [ ] GetCharPressed 能接收到 Unicode 字符
- [ ] 退格键正确处理 UTF-8 多字节字符
- [ ] SDL_TEXTINPUT 事件中 **整段 UTF-8 都入队**（避免只进第一个字）
- [ ] SDL_TEXTEDITING 事件触发并绘制预编辑文本

---

## 中文显示“问号”的修复方案（必须看）

**根因**：字体加载失败或 `TTC` 字体集合未被解析。

### 修复 1：raylib 加入 `.ttc` 支持

raylib 默认只接受 `.ttf/.otf`。需要在 `src/rtext.c` 允许 `.ttc`：

```c
// LoadFontFromMemory
if (TextIsEqual(fileExtLower, ".ttf") ||
    TextIsEqual(fileExtLower, ".otf") ||
    TextIsEqual(fileExtLower, ".ttc"))
{
    ...
}
```

### 修复 2：TTC 需要定位字体集合的偏移

在 `LoadFontData()` 中，初始化字体前尝试 `stbtt_GetFontOffsetForIndex`：

```c
int fontOffset = 0;
if (!stbtt_InitFont(&fontInfo, (unsigned char *)fileData, 0))
{
    int ttcOffset = stbtt_GetFontOffsetForIndex((unsigned char *)fileData, 0);
    if (ttcOffset >= 0) fontOffset = ttcOffset;
}

if (stbtt_InitFont(&fontInfo, (unsigned char *)fileData, fontOffset)) {
    ...
}
```

### 修复 3：确认真正加载到了 CJK 字体

建议输出日志确认：

```c
TraceLog(LOG_INFO, "FONT: Trying %s", fontPath);
TraceLog(LOG_INFO, "FONT: Loaded %s (%d glyphs)", fontPath, font.glyphCount);
```

如果没有 “Loaded” 日志，说明最终仍回退到默认字体（只支持 ASCII）。

---

## 常见踩坑（真实踩过）

1. **SDL3 预编辑事件不触发**：`SDL_HINT_IME_IMPLEMENTED_UI` 必须在 `SDL_Init` 之前设置为 `"composition"`。
2. **SDL3 API 签名变化**：`SDL_StartTextInput(window)` / `SDL_StopTextInput(window)` / `SDL_TextInputActive(window)`。
3. **候选框位置错位**：`GetWindowScaleDPI()` 在 SDL3 上不可靠，改用 `GetRenderWidth/Height` 与 `GetScreenWidth/Height` 比例。
4. **只输入第一个字**：`SDL_TEXTINPUT` 可能一次给多字，必须把整段 UTF-8 全部入队。
5. **字体读到但仍是问号**：需要 `.ttc` 支持 + 字体集合偏移，否则回退默认字体。
6. **双光标**：预编辑状态下要隐藏主光标，仅显示预编辑光标。
7. **预编辑不显示**：必须渲染 `SDL_TEXTEDITING` 文本（SDL 只给数据不绘制）。

---

## 预编辑文本（拼音）显示：完整输入预览

SDL 会通过事件把“预编辑文本（拼音）”发给应用，但**默认不会帮你渲染**。  
想要看到“输入中拼音+下划线”，必须：

### 1) 在 SDL 平台层缓存预编辑信息

文件：`src/platforms/rcore_desktop_sdl.c`

添加结构与状态：

```c
static IMECompositionInfo imeComposition = { 0 };
```

在 `SDL_TEXTEDITING`（SDL2）或 `SDL_EVENT_TEXT_EDITING`（SDL3）里更新：

```c
#if defined(USING_VERSION_SDL3)
case SDL_EVENT_TEXT_EDITING:
    strncpy(imeComposition.text, event.edit.text, sizeof(imeComposition.text) - 1);
    imeComposition.text[sizeof(imeComposition.text) - 1] = '\0';
    imeComposition.cursor = event.edit.start;
    imeComposition.length = event.edit.length;
    break;
#else
case SDL_TEXTEDITING:
    strncpy(imeComposition.text, event.edit.text, sizeof(imeComposition.text) - 1);
    imeComposition.text[sizeof(imeComposition.text) - 1] = '\0';
    imeComposition.cursor = event.edit.start;
    imeComposition.length = event.edit.length;
    break;
#endif
```

暴露 API：

```c
IMECompositionInfo GetIMECompositionInfo(void)
{
    return imeComposition;
}
```

### 2) 在应用侧渲染预编辑文本

```c
IMECompositionInfo comp = GetIMECompositionInfo();
if (comp.text[0] != '\0') {
    DrawTextEx(font, comp.text, caretPos, fontSize, 1.0f, GRAY);
    float w = MeasureTextEx(font, comp.text, fontSize, 1.0f).x;
    DrawLine(caretPos.x, caretPos.y + fontSize,
             caretPos.x + w, caretPos.y + fontSize, GRAY);
}
```

> 你可以选择用更浅的颜色或带下划线，让用户明显看到“预编辑状态”。

## 已知限制

1. **预编辑文本显示**：SDL 使用 "on-the-spot" 模式，预编辑文本（拼音）需要应用自己绘制，需处理 `SDL_TEXTEDITING` 事件
2. **全屏模式**：某些系统下全屏模式可能导致候选框显示异常
3. **平台差异**：不同操作系统的 IME 行为可能略有不同

## 进阶功能（可选）

### 显示预编辑文本（拼音）

如果需要在输入框中显示正在输入的拼音，需要：

1. 在 `rcore.c` 的 `CoreData` 结构体中添加：

```c
struct {
    char compositionText[64];   // 预编辑文本
    int compositionCursor;      // 光标位置
    int compositionLength;      // 选中长度
} IME;
```

2. 在 `SDL_TEXTEDITING` 事件中更新这些值

3. 在应用中读取并显示：

```c
// 绘制预编辑文本（带下划线）
if (strlen(CORE.Input.IME.compositionText) > 0) {
    DrawTextEx(font, CORE.Input.IME.compositionText, cursorPos, 28, 1, GRAY);
    // 绘制下划线
    Vector2 compSize = MeasureTextEx(font, CORE.Input.IME.compositionText, 28, 1);
    DrawLine(cursorPos.x, cursorPos.y + 30, 
             cursorPos.x + compSize.x, cursorPos.y + 30, GRAY);
}
```

## 参考资源

- SDL IME 文档：https://wiki.libsdl.org/SDL2/Tutorials-TextInput
- SDL_SetTextInputRect：https://wiki.libsdl.org/SDL2/SDL_SetTextInputRect
- raylib SDL 后端源码：`src/platforms/rcore_desktop_sdl.c`
- raylib PR #2809（GLFW IME支持尝试）：https://github.com/raysan5/raylib/pull/2809

# Raylib IME 支持实现指南

## 目标

为 raylib 添加中文/日文等 IME（输入法）支持，使输入法候选框能够正确显示在文字输入位置旁边，类似 Chrome 浏览器的行为；并确保 **中文字体能正确加载并渲染**（避免显示 `?`）。

## 技术方案（基于实战结果）

本指南提供 **两个可行路径**，你可以按项目实际选择：

### 方案 A：SDL 后端（跨平台、工程改动在 raylib 平台层）
- 优点：跨平台、候选框位置可控（`SDL_SetTextInputRect`）
- 代价：需要改 raylib 平台层，并切换到 SDL 后端构建

### 方案 B：Windows 原生 IME（不改平台后端、改应用）
- 优点：不需要切后端；在 Win32 下直接可用
- 代价：仅 Windows；需要在应用侧手动设置 IME 窗口位置（`ImmSetCompositionWindow`）

## 前置要求

- raylib 源码（5.0 或更高版本）
- （方案 A）SDL2/SDL3 开发库
- 支持中文的字体文件（建议：`NotoSansCJK-Regular.ttc` 或 `NotoSansCJK-Regular.otf`）

## 实现步骤

> **重要经验**：中文显示成 `?` 的常见原因不是 IME，而是字体没真正加载成功，或 `TTC` 字体集合未被解析。

### 第一步：修改 SDL 后端源文件

文件路径：`src/platforms/rcore_desktop_sdl.c`

#### 1.1 在文件顶部添加 IME 相关变量

在 `PlatformData` 结构体或全局变量区域添加：

```c
// IME Support
static bool imeEnabled = false;
static SDL_Rect imeRect = { 0, 0, 1, 20 };
```

#### 1.2 修改 InitPlatform() 函数

在 `SDL_CreateWindow()` 调用**之前**添加 IME hint：

```c
// Enable native IME UI (candidate window)
SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");

// 对于 SDL3，还可以设置：
// SDL_SetHint(SDL_HINT_IME_INTERNAL_EDITING, "1");
```

查找 `SDL_CreateWindow` 调用位置，在其前面插入上述代码。

#### 1.3 添加 IME 控制函数

在文件末尾（`// Module Functions Definition` 区域）添加以下函数：

```c
//----------------------------------------------------------------------------------
// Module Functions Definition: IME Support
//----------------------------------------------------------------------------------

// Start text input mode (activate IME)
void StartTextInput(void)
{
    SDL_StartTextInput();
    imeEnabled = true;
}

// Stop text input mode (deactivate IME)
void StopTextInput(void)
{
    SDL_StopTextInput();
    imeEnabled = false;
}

// Check if text input is active
bool IsTextInputActive(void)
{
    return SDL_IsTextInputActive();
}

// Set IME candidate window position
// Call this every frame with the current text cursor position
void SetTextInputRect(int x, int y, int width, int height)
{
    imeRect.x = x;
    imeRect.y = y;
    imeRect.w = width;
    imeRect.h = height;
    SDL_SetTextInputRect(&imeRect);
}

// Get current IME rect
Rectangle GetTextInputRect(void)
{
    return (Rectangle){ (float)imeRect.x, (float)imeRect.y, 
                        (float)imeRect.w, (float)imeRect.h };
}
```

#### 1.4 关键点：候选框位置的坐标系（必须处理 DPI/缩放）

`SDL_SetTextInputRect` 需要的是**窗口客户端区域的像素坐标**，而不是 world 坐标或缩放后的虚拟坐标。  
如果你用了高 DPI、摄像机（`BeginMode2D`）、或 RenderTexture 做缩放/信箱渲染，必须把光标位置转换到**屏幕像素**再传入：

```c
// 1) 得到光标在屏幕空间的坐标 (screen-space, 逻辑像素)
Vector2 caretScreen = caretPos; // 默认就是 screen-space
// 如果在 2D 摄像机下绘制 UI，先转到屏幕坐标：
// Vector2 caretScreen = GetWorldToScreen2D(caretWorldPos, camera);

// 2) 处理高 DPI：转换到窗口像素
Vector2 dpi = GetWindowScaleDPI();   // 例如 1.0/1.5/2.0
int imeX = (int)(caretScreen.x * dpi.x + 0.5f);
int imeY = (int)((caretScreen.y + lineHeight) * dpi.y + 0.5f);
int imeH = (int)(lineHeight * dpi.y + 0.5f);

// 3) 设置候选框位置（每帧更新）
SetTextInputRect(imeX, imeY, 1, imeH);
```

> 经验：候选框通常显示在**光标矩形的右下方**，所以把 `y` 设为 `caretY + lineHeight` 更稳。

#### 1.5 处理 SDL_TEXTEDITING 事件（可选，用于显示预编辑文本）

在 `PollInputEvents()` 函数中找到 `SDL_TEXTINPUT` 事件处理，在其附近添加：

```c
case SDL_TEXTEDITING:
{
    // event.edit.text  - 当前预编辑文本（如拼音）
    // event.edit.start - 光标位置
    // event.edit.length - 选中长度
    
    // 存储预编辑信息供应用使用
    // 注意：这需要在 CORE 结构中添加相应字段
    // CORE.Input.IME.compositionText
    // CORE.Input.IME.compositionCursor
    // CORE.Input.IME.compositionLength
    
    TRACELOG(LOG_DEBUG, "IME Composition: %s", event.edit.text);
} break;
```

### 第二步：修改 raylib.h 头文件

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

### 第三步：编译 raylib（使用 SDL 后端）

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

CMake 目前不直接支持 SDL 后端，需要修改 CMakeLists.txt 或使用 Makefile。

### 第四步：创建测试示例

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
    Vector2 dpi = GetWindowScaleDPI();
    int imeX = (int)(caret.x * dpi.x + 0.5f);
    int imeY = (int)((caret.y + fontSize) * dpi.y + 0.5f);
    int imeH = (int)(fontSize * dpi.y + 0.5f);
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

- [ ] SDL_SetHint 在 SDL_CreateWindow 之前调用
- [ ] StartTextInput/StopTextInput 正确控制 IME 状态
- [ ] SetTextInputRect 每帧更新候选框位置
- [ ] 已将光标位置转换到窗口像素坐标（含 DPI/缩放/摄像机）
- [ ] 中文字体正确加载
- [ ] GetCharPressed 能接收到 Unicode 字符
- [ ] 退格键正确处理 UTF-8 多字节字符

---

## 方案 B：Windows 原生 IME（应用侧修改，不切后端）

> 适用于：你使用 raylib 默认 Win32/GLFW 后端，不想切 SDL，但需要 IME 候选框跟随光标。

### B1. 在应用侧引入 Win32 IME API

```c
#if defined(_WIN32)
    #include <windows.h>
    #include <imm.h>
#endif
```

### B2. 设置候选框位置（每帧更新）

```c
#if defined(_WIN32)
static void UpdateImeWindowPosition(int x, int y) {
    HWND hwnd = (HWND)GetWindowHandle(); // raylib API
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
#endif
```

调用位置（每帧，或光标移动后）：

```c
Vector2 dpi = GetWindowScaleDPI();
int imeX = (int)(caretX * dpi.x + 0.5f);
int imeY = (int)((caretY + lineHeight) * dpi.y + 0.5f);
UpdateImeWindowPosition(imeX, imeY);
```

### B3. 编译链接

Windows 需要链接 `imm32`：

- MSVC/CMake:
  - `target_link_libraries(your_target PRIVATE imm32)`
- MinGW:
  - `-limm32`

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

## 常见踩坑

1. **只看到 FILEIO 成功**：`LoadFileData` 读到文件不代表字体解析成功。要看 `FONT: Data loaded successfully`。
2. **TTC 文件未被识别**：必须加 `.ttc` 扩展支持，否则默认回退。
3. **候选框位置不对**：必须用窗口像素坐标（考虑 DPI/缩放）。
4. **预编辑文本显示为空**：需要处理 `SDL_TEXTEDITING`（方案 A）或自己渲染预编辑文本。

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

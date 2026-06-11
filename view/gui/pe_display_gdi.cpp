/**
 * pe_display_gdi.c — Win32 GDI display layer
 *
 * Pure GDI replacement for PainterEngine's D2D display.
 * Implements the interface defined in platform/windows/px_display.h
 * so that platform/windows/px_main.c works without Direct2D.
 */

#include <Windows.h>
#include <WindowsX.h>
#include <stdio.h>
#include <time.h>

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Gdi32.lib")
#pragma comment(lib, "Winmm.lib")
#pragma comment(lib, "Shell32.lib")
#pragma comment(lib, "Imm32.lib")
#pragma comment(lib, "Msimg32.lib")

typedef struct {
    UINT uMsg;
    LPARAM lparam;
    WPARAM wparam;
} WM_MESSAGE;

typedef enum {
    PX_WINODW_STYLE_NORMAL,
    PX_WINODW_STYLE_DESKTOPSPRITE,
    PX_WINODW_STYLE_SIMPLEWINDOW,
    PX_WINODW_STYLE_FIXED,
    PX_WINODW_STYLE_NOBOARD,
} PX_WINODW_STYLE;

static HWND  g_hwnd = NULL;
static int   g_surface_w = 0, g_surface_h = 0;
static int   g_window_w = 0, g_window_h = 0;
static HDC   g_memDC = NULL;
static HBITMAP g_memBmp = NULL;
static void *g_memBits = NULL;
static BOOL  g_activated = TRUE;
static BOOL  g_lDown = FALSE, g_rDown = FALSE, g_mDown = FALSE;
static int   g_mouseX = 0, g_mouseY = 0;
static BOOL  g_keyDown[256] = {0};
static char  g_dragFile[MAX_PATH] = {0};
static int   g_wheelX = 0, g_wheelY = 0, g_wheelDelta = 0;
static int   g_hasWheel = 0;
static double g_scale = 1.0;

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        if (g_memDC && g_memBits) {
            BITMAPINFOHEADER bmih = {0};
            bmih.biSize = sizeof(BITMAPINFOHEADER);
            bmih.biWidth = g_surface_w;
            bmih.biHeight = -g_surface_h;
            bmih.biPlanes = 1;
            bmih.biBitCount = 32;
            bmih.biCompression = BI_RGB;
            SetDIBitsToDevice(hdc, 0, 0, g_surface_w, g_surface_h,
                             0, 0, 0, g_surface_h, g_memBits,
                             (BITMAPINFO*)&bmih, DIB_RGB_COLORS);
        }
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_ACTIVATE:
        g_activated = (LOWORD(wp) != WA_INACTIVE);
        break;
    case WM_LBUTTONDOWN: g_lDown = TRUE; g_mouseX = GET_X_LPARAM(lp); g_mouseY = GET_Y_LPARAM(lp); break;
    case WM_LBUTTONUP:   g_lDown = FALSE; break;
    case WM_RBUTTONDOWN: g_rDown = TRUE; g_mouseX = GET_X_LPARAM(lp); g_mouseY = GET_Y_LPARAM(lp); break;
    case WM_RBUTTONUP:   g_rDown = FALSE; break;
    case WM_MBUTTONDOWN: g_mDown = TRUE; g_mouseX = GET_X_LPARAM(lp); g_mouseY = GET_Y_LPARAM(lp); break;
    case WM_MBUTTONUP:   g_mDown = FALSE; break;
    case WM_MOUSEMOVE:   g_mouseX = GET_X_LPARAM(lp); g_mouseY = GET_Y_LPARAM(lp); break;
    case WM_KEYDOWN:     if (wp < 256) g_keyDown[wp] = TRUE; break;
    case WM_KEYUP:       if (wp < 256) g_keyDown[wp] = FALSE; break;
    case WM_MOUSEWHEEL:  g_hasWheel = 1; g_wheelDelta = GET_WHEEL_DELTA_WPARAM(wp);
                         g_mouseX = GET_X_LPARAM(lp); g_mouseY = GET_Y_LPARAM(lp); break;
    case WM_DROPFILES: {
        HDROP hDrop = (HDROP)wp;
        if (DragQueryFileA(hDrop, 0, g_dragFile, MAX_PATH) > 0) {}
        DragFinish(hDrop);
        break;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

static void CreateMemBitmap(void) {
    HDC hdc = GetDC(g_hwnd);
    if (g_memDC) DeleteDC(g_memDC);
    if (g_memBmp) DeleteObject(g_memBmp);

    g_memDC = CreateCompatibleDC(hdc);
    BITMAPINFO bmi = {0};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = g_surface_w;
    bmi.bmiHeader.biHeight = -g_surface_h;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    g_memBmp = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &g_memBits, NULL, 0);
    SelectObject(g_memDC, g_memBmp);
    ReleaseDC(g_hwnd, hdc);
}

extern "C" {

BOOL PX_CreateWindow(int surfaceWidth, int surfaceHeight, int windowWidth, int windowHeight, const char *name) {
    static int registered = 0;
    if (!registered) {
        WNDCLASSA wc = {0};
        wc.lpfnWndProc = WndProc;
        wc.hInstance = GetModuleHandleA(NULL);
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        wc.lpszClassName = "PainterEngineWnd";
        RegisterClassA(&wc);
        registered = 1;
    }

    RECT rc = {0, 0, windowWidth, windowHeight};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    g_surface_w = surfaceWidth;
    g_surface_h = surfaceHeight;
    g_window_w = rc.right - rc.left;
    g_window_h = rc.bottom - rc.top;

    g_hwnd = CreateWindowA("PainterEngineWnd", name, WS_OVERLAPPEDWINDOW,
                           CW_USEDEFAULT, CW_USEDEFAULT,
                           g_window_w, g_window_h,
                           NULL, NULL, GetModuleHandleA(NULL), NULL);
    if (!g_hwnd) return FALSE;

    DragAcceptFiles(g_hwnd, TRUE);
    CreateMemBitmap();
    ShowWindow(g_hwnd, SW_SHOW);
    UpdateWindow(g_hwnd);
    return TRUE;
}

BOOL PX_WindowResize(int surfaceWidth, int surfaceHeight, int windowWidth, int windowHeight) {
    g_surface_w = surfaceWidth;
    g_surface_h = surfaceHeight;
    CreateMemBitmap();
    return TRUE;
}

int  PX_GetScreenHeight() { return GetSystemMetrics(SM_CYSCREEN); }
int  PX_GetScreenWidth()  { return GetSystemMetrics(SM_CXSCREEN); }
int  PX_SetWindowResizeable() { return 0; }
HWND PX_GetWindowHwnd() { return g_hwnd; }

VOID PX_SystemReadDeviceState() {}

BOOL PX_SystemLoop() {
    MSG msg;
    while (PeekMessageA(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) return FALSE;
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return TRUE;
}

BOOL PX_SystemRender(void *raw, int width, int height) {
    if (!g_memBits || !raw) return FALSE;
    memcpy(g_memBits, raw, width * height * 4);
    InvalidateRect(g_hwnd, NULL, FALSE);
    UpdateWindow(g_hwnd);
    return TRUE;
}

BOOL PX_SystemisAvtivated() { return g_activated; }
BOOL PX_KeyboardDown(unsigned char X) { return (X < 256) ? g_keyDown[X] : FALSE; }

const char *PX_KeyboardString() { return NULL; }
char *PX_DragfileString() { return g_dragFile[0] ? g_dragFile : NULL; }

BOOL PX_MouseLButtonDown() { return g_lDown; }
BOOL PX_MouseRButtonDown() { return g_rDown; }
BOOL PX_MouseMButtonDown() { return g_mDown; }
POINT PX_MousePosition() { POINT p = {g_mouseX, g_mouseY}; return p; }
BOOL PX_KeyDown(unsigned char key) { return (key < 256) ? g_keyDown[key] : FALSE; }

BOOL PX_MouseWheel(int *x, int *y, int *delta) {
    if (!g_hasWheel) return FALSE;
    g_hasWheel = 0;
    if (x) *x = g_mouseX;
    if (y) *y = g_mouseY;
    if (delta) *delta = g_wheelDelta;
    return TRUE;
}

BOOL PX_GetWinMessage(WM_MESSAGE *Msg) { return FALSE; }
double PX_GetWindowScale() { return g_scale; }

void PX_SetWindowStyle(PX_WINODW_STYLE style) {
    if (!g_hwnd) return;
    LONG ws = WS_OVERLAPPEDWINDOW;
    if (style == PX_WINODW_STYLE_FIXED)
        ws = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;
    SetWindowLongA(g_hwnd, GWL_STYLE, ws);
    SetWindowPos(g_hwnd, NULL, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED);
}

void PX_SetWindowPosition(int x, int y) { if (g_hwnd) SetWindowPos(g_hwnd, NULL, x, y, 0, 0, SWP_NOSIZE); }
void PX_SetWindowName(const char* name) { if (g_hwnd) SetWindowTextA(g_hwnd, name); }

}

extern "C" int PX_AudioInitialize(void *soundplay) { (void)soundplay; return 1; }
extern "C" int PX_AudioInitializeHwnd(void *hwnd) { (void)hwnd; return 1; }

extern "C" unsigned int PX_TimeGetTime() { return (unsigned int)timeGetTime(); }
extern "C" unsigned long long PX_Time() {
    FILETIME ft; GetSystemTimeAsFileTime(&ft);
    return ((unsigned long long)ft.dwHighDateTime << 32 | ft.dwLowDateTime) / 10000;
}
extern "C" void PX_Sleep(unsigned int ms) { Sleep(ms); }
extern "C" unsigned int PX_TimeGetTimeUs() {
    static LARGE_INTEGER freq = {0}, counter;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (unsigned int)(counter.QuadPart * 1000000LL / freq.QuadPart);
}
extern "C" struct px_timem { int year,month,day,hour,minute,second; };
extern "C" px_timem PX_LocalTime(long long timet) {
    time_t t = (time_t)timet;
    struct tm *p = gmtime(&t);
    px_timem r = {0};
    if (p) { r.year=1900+p->tm_year; r.month=1+p->tm_mon; r.day=p->tm_mday;
             r.hour=p->tm_hour; r.minute=p->tm_min; r.second=p->tm_sec; }
    return r;
}
extern "C" int PX_TimeGetYear()   { time_t t=time(NULL); struct tm *p=gmtime(&t); return p?1900+p->tm_year:0; }
extern "C" int PX_TimeGetMonth()  { time_t t=time(NULL); struct tm *p=gmtime(&t); return p?1+p->tm_mon:0; }
extern "C" int PX_TimeGetDay()    { time_t t=time(NULL); struct tm *p=gmtime(&t); return p?p->tm_mday:0; }
extern "C" int PX_TimeGetHour()   { time_t t=time(NULL); struct tm *p=gmtime(&t); return p?p->tm_hour:0; }
extern "C" int PX_TimeGetMinute() { time_t t=time(NULL); struct tm *p=gmtime(&t); return p?p->tm_min:0; }
extern "C" int PX_TimeGetSecond() { time_t t=time(NULL); struct tm *p=gmtime(&t); return p?p->tm_sec:0; }

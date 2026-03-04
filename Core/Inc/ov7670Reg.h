/*
 * ov7670Reg.h
 *
 *  Created on: 2017/08/25
 *      Author: take-iwiw
 */

#ifndef OV7670_OV7670REG_H_
#define OV7670_OV7670REG_H_

#define REG_BATT 0xFF

// const uint8_t OV7670_reg[][2] = {
//     /* Color mode related */
//     {0x12, 0x14},        // QVGA, RGB
//     {0x8C, 0x00},        // RGB444 Disable
//     {0x40, 0x10 + 0xc0}, // RGB565, 00 - FF
//     {0x3A, 0x04 + 8},    // UYVY (why?)
//     {0x3D, 0x80 + 0x00}, // gamma enable, UV auto adjust, UYVY
//     {0xB0, 0x84},        // important

//     /* clock related */
//     {0x0C, 0x04}, // DCW enable
//     {0x3E, 0x19}, // manual scaling, pclk/=2
//     {0x70, 0x3A}, // scaling_xsc
//     {0x71, 0x35}, // scaling_ysc
//     {0x72, 0x11}, // down sample by 2
//     {0x73, 0xf1}, // DSP clock /= 2

//     /* windowing (empirically decided...) */
//     {0x17, 0x16}, // HSTART
//     {0x18, 0x04}, // HSTOP
//     {0x32, 0x80}, // HREF
//     {0x19, 0x03}, // VSTART =  14 ( = 3 * 4 + 2)
//     {0x1a, 0x7b}, // VSTOP  = 494 ( = 123 * 4 + 2)
//     {0x03, 0x0a}, // VREF (VSTART_LOW = 2, VSTOP_LOW = 2)

// /* color matrix coefficient */
// #if 0
//   {0x4f, 0xb3},
//   {0x50, 0xb3},
//   {0x51, 0x00},
//   {0x52, 0x3d},
//   {0x53, 0xa7},
//   {0x54, 0xe4},
//   {0x58, 0x9e},
// #else
//     {0x4f, 0x80},
//     {0x50, 0x80},
//     {0x51, 0x00},
//     {0x52, 0x22},
//     {0x53, 0x5e},
//     {0x54, 0x80},
//     {0x58, 0x9e},
// #endif

//     /* 3a */
//     //  {0x13, 0x84},
//     //  {0x14, 0x0a},   // AGC Ceiling = 2x
//     //  {0x5F, 0x2f},   // AWB B Gain Range (empirically decided)
//     //                  // without this bright scene becomes yellow (purple).
//     //                  might be because of color matrix
//     //  {0x60, 0x98},   // AWB R Gain Range (empirically decided)
//     //  {0x61, 0x70},   // AWB G Gain Range (empirically decided)
//     {0x41, 0x38}, // edge enhancement, de-noise, AWG gain enabled

// /* gamma curve */
// #if 1
//     {0x7b, 16},
//     {0x7c, 30},
//     {0x7d, 53},
//     {0x7e, 90},
//     {0x7f, 105},
//     {0x80, 118},
//     {0x81, 130},
//     {0x82, 140},
//     {0x83, 150},
//     {0x84, 160},
//     {0x85, 180},
//     {0x86, 195},
//     {0x87, 215},
//     {0x88, 230},
//     {0x89, 244},
//     {0x7a, 16},
// #else
//     /* gamma = 1 */
//     {0x7b, 4},
//     {0x7c, 8},
//     {0x7d, 16},
//     {0x7e, 32},
//     {0x7f, 40},
//     {0x80, 48},
//     {0x81, 56},
//     {0x82, 64},
//     {0x83, 72},
//     {0x84, 80},
//     {0x85, 96},
//     {0x86, 112},
//     {0x87, 144},
//     {0x88, 176},
//     {0x89, 208},
//     {0x7a, 64},
// #endif

//     /* fps */
//     //  {0x6B, 0x4a}, //PLL  x4
//     {0x11, 0x00}, // pre-scalar = 1/1

//     /* others */
//     {0x1E, 0x31}, // mirror flip
//     //  {0x42, 0x08}, // color bar

//     {REG_BATT, REG_BATT},
// };

// by clarobit
const uint8_t OV7670_reg_manual[][2] = {

    /* ===== Format: RGB565 (clean) ===== */
    {0x12, 0x04}, // COM7: RGB mode
    {0x8C, 0x00}, // RGB444 disable
    {0x40, 0xD0}, // COM15: RGB565 + full range
    {0x3A, 0x00}, // TSLB: RGB byte order (disable UYVY related bits)
    {0x3D, 0x00}, // COM13: disable UV swap/auto (RGB 안정화)

    /* ===== Clock ===== */
    {0x11, 0x00}, // CLKRC: prescaler = 1
    {0x3E, 0x19}, // COM14: manual scaling, PCLK /= 2

    /* ===== Scaling (QVGA) ===== */
    {0x0C, 0x04}, // COM3: DCW enable
    {0x70, 0x3A}, // SCALING_XSC
    {0x71, 0x35}, // SCALING_YSC
    {0x72, 0x11}, // SCALING_DCWCTR (downsample by 2)
    {0x73, 0xF1}, // SCALING_PCLK_DIV

    /* ===== Windowing (QVGA) ===== */
    {0x17, 0x16}, // HSTART
    {0x18, 0x04}, // HSTOP
    {0x32, 0x80}, // HREF
    {0x19, 0x03}, // VSTART
    {0x1A, 0x7B}, // VSTOP
    {0x03, 0x0A}, // VREF

    /* ===== Automatic control (실영상 색 자연화) ===== */
    {0x13, 0xE0}, // COM8: AWB + AGC + AEC enable
    {0x13, 0xE7}, // COM8: AWB + AGC + AEC enable
    {0x14, 0x48}, // AGC ceiling
    {0x5F, 0x2F}, // AWB B gain range
    // {0x60, 0x98}, // AWB R gain range
    {0x60, 0x70}, // AWB R gain range 더 ↓
    // {0x61, 0x70}, // AWB G gain range
    {0x61, 0x60}, // G 약간 감소 (최종 권장)

    /* ===== Gamma curve ===== */
    {0x7B, 16},
    {0x7C, 30},
    {0x7D, 53},
    {0x7E, 90},
    {0x7F, 105},
    {0x80, 118},
    {0x81, 130},
    {0x82, 140},
    {0x83, 150},
    {0x84, 160},
    {0x85, 180},
    {0x86, 195},
    {0x87, 215},
    {0x88, 230},
    {0x89, 244},
    {0x7A, 16},

    /* ===== Color matrix (필요 시 조정) ===== */
    {0x4F, 0x80},
    {0x50, 0x80},
    {0x51, 0x00},
    {0x52, 0x22},
    {0x53, 0x5E},
    {0x54, 0x80},
    {0x58, 0x9E},

    /* ===== Edge / denoise ===== */
    {0x41, 0x38}, // edge enhancement + denoise

    /* ===== Mirror / flip (필요 없으면 주석) ===== */
    {0x1E, 0x31},

    /* ===== Test pattern OFF ===== */
    // {0x42, 0x08},   // Color bar enable
    {0x42, 0x00}, // color bar OFF

    /* End marker */
    {REG_BATT, REG_BATT},
};

#endif /* OV7670_OV7670REG_H_ */
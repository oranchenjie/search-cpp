#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace std::chrono;

// ==============================================
// 【核心硬约束】严格300字节，丝毫不差
// ==============================================
const int TOTAL_PACKET_BYTE = 300;
const int HEADER_BYTE = 6;       // 帧头6字节
const int BALL_DATA_BYTE = 8;     // 2个弹丸，8字节
const int VECTOR_DATA_MAX_BYTE = TOTAL_PACKET_BYTE - HEADER_BYTE - BALL_DATA_BYTE; // 286字节
static_assert(HEADER_BYTE + BALL_DATA_BYTE + VECTOR_DATA_MAX_BYTE == TOTAL_PACKET_BYTE, "300字节硬约束校验失败");

// 原始视频分辨率
const Size ORIGINAL_SIZE = Size(1350, 1080);
// 坐标优化：9bit=0-511
const int COORD_BITS = 9;
const int COORD_MAX = (1 << COORD_BITS) - 1; // 511
const double SCALE_FACTOR = (double)COORD_MAX / max(ORIGINAL_SIZE.width, ORIGINAL_SIZE.height);
// 单条线段优化后占用：4.5字节，两条打包成9字节
const int LINE_PAIR_BYTE_SIZE = 9;
const int MAX_LINE_COUNT = 71;
// 弹丸轨迹ROI范围
const int BALL_TRAJECTORY_ROI = 150;

// ===================== 配置位定义 =====================
#define CONFIG_FRAME_VALID     0x01
#define CONFIG_MODE_VECTOR     0x02
#define CONFIG_BALL_VALID      0x04

// ===================== 弹丸识别参数 =====================
const float MIN_BALL_AREA = 100.0f;
const float MAX_BALL_AREA = 2000.0f;
const float MIN_BALL_CIRCULARITY = 0.8f;
const float MAX_BALL_ASPECT_RATIO = 1.3f;
const Scalar BALL_HSV_LOW = Scalar(35, 8, 140);
const Scalar BALL_HSV_HIGH = Scalar(100, 255, 255);

// ===================== 霍夫直线检测参数 =====================
const float HOUGH_RHO = 1.0f;
const float HOUGH_THETA = CV_PI / 180.0f;
const int HOUGH_THRESHOLD = 20;
const double HOUGH_MIN_LINE_LENGTH = 5.0;
const double HOUGH_MAX_LINE_GAP = 10.0;

// ===================== 300字节数据包结构 =====================
#pragma pack(1)
struct MqttPacket {
    uint8_t frame_seq;
    uint8_t config;
    uint16_t width;
    uint16_t height;
    uint8_t ball_data[BALL_DATA_BYTE];
    uint8_t vector_data[VECTOR_DATA_MAX_BYTE];
};
#pragma pack()
static_assert(sizeof(MqttPacket) == TOTAL_PACKET_BYTE, "数据包必须严格300字节");

// ===================== 处理结果结构体 =====================
struct ProcessResult
{
    cv::Mat originalMarked;
    MqttPacket packet;
    int line_count;
    int ballCount;
    vector<Vec4i> lines;
    vector<Point2f> ballCenters;
    vector<float> ballRadii;
    int data_used_byte;
};

// ===================== 工具函数：坐标量化/反量化 =====================
inline uint16_t quantizeCoord(int val) {
    return (uint16_t)round(val * SCALE_FACTOR);
}
inline int dequantizeCoord(uint16_t val) {
    return (int)round(val / SCALE_FACTOR);
}

// ===================== 弹丸打包/解包函数 =====================
inline void packBall(uint8_t* buf, Point center, int radius, bool valid) {
    buf[0] = buf[1] = buf[2] = buf[3] = 0;
    if (!valid) return;

    uint16_t x = quantizeCoord(center.x);
    uint16_t y = quantizeCoord(center.y);
    uint8_t r = (uint8_t)min(radius, 63);

    buf[0] = (x >> 1) & 0xFF;
    buf[1] = ((x & 0x01) << 7) | ((y >> 2) & 0x7F);
    buf[2] = ((y & 0x03) << 6) | ((r >> 2) & 0x3F);
    buf[3] = ((r & 0x03) << 6) & 0xC0;
}

inline void unpackBall(const uint8_t* buf, Point& center, int& radius, bool& valid) {
    uint16_t x = ((buf[0] & 0xFF) << 1) | ((buf[1] >> 7) & 0x01);
    uint16_t y = ((buf[1] & 0x7F) << 2) | ((buf[2] >> 6) & 0x03);
    uint8_t r = ((buf[2] & 0x3F) << 2) | ((buf[3] >> 6) & 0x03);

    center.x = dequantizeCoord(x);
    center.y = dequantizeCoord(y);
    radius = r;
    valid = (x != 0 || y != 0);
}

// ===================== 两条线段打包成9字节 =====================
inline void packLinePair(uint8_t* buf, Vec4i line1, Vec4i line2) {
    uint16_t x1 = quantizeCoord(line1[0]), y1 = quantizeCoord(line1[1]);
    uint16_t x2 = quantizeCoord(line1[2]), y2 = quantizeCoord(line1[3]);
    uint16_t x3 = quantizeCoord(line2[0]), y3 = quantizeCoord(line2[1]);
    uint16_t x4 = quantizeCoord(line2[2]), y4 = quantizeCoord(line2[3]);

    buf[0] = (x1 >> 1) & 0xFF;
    buf[1] = ((x1 & 0x01) << 7) | ((y1 >> 2) & 0x7F);
    buf[2] = ((y1 & 0x03) << 6) | ((x2 >> 3) & 0x3F);
    buf[3] = ((x2 & 0x07) << 5) | ((y2 >> 4) & 0x1F);
    buf[4] = ((y2 & 0x0F) << 4) | ((x3 >> 5) & 0x0F);
    buf[5] = ((x3 & 0x1F) << 3) | ((y3 >> 6) & 0x07);
    buf[6] = ((y3 & 0x3F) << 2) | ((x4 >> 7) & 0x03);
    buf[7] = ((x4 & 0x7F) << 1) | ((y4 >> 8) & 0x01);
    buf[8] = y4 & 0xFF;
}

inline void unpackLinePair(const uint8_t* buf, Vec4i& line1, Vec4i& line2) {
    uint16_t x1 = ((buf[0] & 0xFF) << 1) | ((buf[1] >> 7) & 0x01);
    uint16_t y1 = ((buf[1] & 0x7F) << 2) | ((buf[2] >> 6) & 0x03);
    uint16_t x2 = ((buf[2] & 0x3F) << 3) | ((buf[3] >> 5) & 0x07);
    uint16_t y2 = ((buf[3] & 0x1F) << 4) | ((buf[4] >> 4) & 0x0F);
    uint16_t x3 = ((buf[4] & 0x0F) << 5) | ((buf[5] >> 3) & 0x1F);
    uint16_t y3 = ((buf[5] & 0x07) << 6) | ((buf[6] >> 2) & 0x3F);
    uint16_t x4 = ((buf[6] & 0x03) << 7) | ((buf[7] >> 1) & 0x7F);
    uint16_t y4 = ((buf[7] & 0x01) << 8) | (buf[8] & 0xFF);

    line1 = Vec4i(dequantizeCoord(x1), dequantizeCoord(y1), dequantizeCoord(x2), dequantizeCoord(y2));
    line2 = Vec4i(dequantizeCoord(x3), dequantizeCoord(y3), dequantizeCoord(x4), dequantizeCoord(y4));
}

// ===================== 【修正】判断线段是否在弹丸轨迹ROI内 =====================
bool isLineInBallROI(Vec4i line, const vector<Point2f>& ballCenters, int roi_range) {
    if (ballCenters.empty()) return false;
    // 修正：统一转为Point2f类型，解决类型不匹配问题
    Point2f p1((float)line[0], (float)line[1]);
    Point2f p2((float)line[2], (float)line[3]);
    for (const auto& center : ballCenters) {
        // 修正：同类型点相减，norm可以正常计算距离
        if (cv::norm(p1 - center) < roi_range || cv::norm(p2 - center) < roi_range) {
            return true;
        }
    }
    return false;
}

// ===================== 核心处理器类 =====================
class HeroCamCompressor
{
public:
    ProcessResult process(Mat &input)
    {
        ProcessResult result;
        if (input.empty()) return result;
        int origW = input.cols;
        int origH = input.rows;

        // 1. 边缘提取
        Mat gray, blurred, edges;
        cvtColor(input, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurred, Size(5, 5), 1.3);
        Canny(blurred, edges, 50, 150);
        
        // 2. 霍夫直线变换
        vector<Vec4i> all_lines;
        HoughLinesP(edges, all_lines, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);

        // 3. 弹丸识别
        Mat hsv, greenMask;
        cvtColor(input, hsv, COLOR_BGR2HSV);
        inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH, greenMask);
        
        Mat kernel = getStructuringElement(MORPH_RECT, Size(2,2));
        morphologyEx(greenMask, greenMask, MORPH_CLOSE, kernel);
        dilate(greenMask, greenMask, kernel);

        vector<vector<Point>> ballContours;
        findContours(greenMask.clone(), ballContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        sort(ballContours.begin(), ballContours.end(), [](const vector<Point>& a, const vector<Point>& b) {
            return contourArea(a) > contourArea(b);
        });

        int validBalls = 0;
        Point ball1(0,0), ball2(0,0);
        int r1=0, r2=0;
        bool valid1=false, valid2=false;

        for (const auto &cnt : ballContours)
        {
            double area = contourArea(cnt);
            if (area < MIN_BALL_AREA || area > MAX_BALL_AREA) continue;

            double perim = arcLength(cnt, true);
            if (perim <= 0) continue;
            double circularity = 4.0 * CV_PI * area / (perim * perim);
            if (circularity < MIN_BALL_CIRCULARITY) continue;

            Rect rect = boundingRect(cnt);
            double aspect = static_cast<double>(rect.width) / rect.height;
            if (aspect < 1.0) aspect = 1.0 / aspect;
            if (aspect > MAX_BALL_ASPECT_RATIO) continue;

            Point2f center;
            float radius;
            minEnclosingCircle(cnt, center, radius);
            result.ballCenters.push_back(center);
            result.ballRadii.push_back(radius);

            if (!valid1) { ball1 = Point(cvRound(center.x), cvRound(center.y)); r1 = cvRound(radius); valid1 = true; }
            else if (!valid2) { ball2 = Point(cvRound(center.x), cvRound(center.y)); r2 = cvRound(radius); valid2 = true; }
            validBalls++;
        }
        result.ballCount = validBalls;

        // ==============================================
        // 线段优先级：轨迹线段 > 场地长线段
        // ==============================================
        vector<Vec4i> trajectory_lines;
        vector<Vec4i> field_lines;

        for (const auto& line : all_lines) {
            if (isLineInBallROI(line, result.ballCenters, BALL_TRAJECTORY_ROI)) {
                trajectory_lines.push_back(line);
            } else {
                field_lines.push_back(line);
            }
        }

        sort(field_lines.begin(), field_lines.end(), [](const Vec4i& a, const Vec4i& b) {
            double lenA = cv::norm(Point(a[0], a[1]) - Point(a[2], a[3]));
            double lenB = cv::norm(Point(b[0], b[1]) - Point(b[2], b[3]));
            return lenA > lenB;
        });

        vector<Vec4i> final_lines;
        for (const auto& line : trajectory_lines) {
            if (final_lines.size() >= MAX_LINE_COUNT) break;
            final_lines.push_back(line);
        }
        for (const auto& line : field_lines) {
            if (final_lines.size() >= MAX_LINE_COUNT) break;
            final_lines.push_back(line);
        }

        result.lines = final_lines;
        result.line_count = final_lines.size();

        // 生成左侧原始画面
        Mat originalMarked = Mat::zeros(input.size(), CV_8UC3);
        for (const auto& l : all_lines) {
            cv::line(originalMarked, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 2, LINE_AA);
        }
        for (size_t i = 0; i < result.ballCenters.size(); i++) {
            cv::circle(originalMarked, result.ballCenters[i], (int)result.ballRadii[i], Scalar(255, 255, 255), -1, LINE_AA);
            cv::circle(originalMarked, result.ballCenters[i], (int)result.ballRadii[i] + 3, Scalar(0, 255, 0), 3, LINE_AA);
        }
        result.originalMarked = originalMarked;

        // ==============================================
        // 打包300字节数据包
        // ==============================================
        MqttPacket pkt;
        memset(&pkt, 0, sizeof(MqttPacket));
        pkt.config = CONFIG_FRAME_VALID;
        pkt.width = origW;
        pkt.height = origH;

        packBall(pkt.ball_data, ball1, r1, valid1);
        packBall(pkt.ball_data + 4, ball2, r2, valid2);
        if (valid1 || valid2) pkt.config |= CONFIG_BALL_VALID;

        memset(pkt.vector_data, 0, VECTOR_DATA_MAX_BYTE);
        int line_count = final_lines.size();
        int pair_count = line_count / 2;
        int single_line = line_count % 2;
        
        for (int i = 0; i < pair_count; i++) {
            int offset = i * LINE_PAIR_BYTE_SIZE;
            if (offset + LINE_PAIR_BYTE_SIZE > VECTOR_DATA_MAX_BYTE) break;
            packLinePair(pkt.vector_data + offset, final_lines[i*2], final_lines[i*2+1]);
        }
        
        if (single_line && pair_count * LINE_PAIR_BYTE_SIZE + 5 <= VECTOR_DATA_MAX_BYTE) {
            Vec4i dummy_line(0,0,0,0);
            packLinePair(pkt.vector_data + pair_count * LINE_PAIR_BYTE_SIZE, final_lines[line_count-1], dummy_line);
        }

        int data_used = 0;
        for (int i = VECTOR_DATA_MAX_BYTE - 1; i >= 0; i--) {
            if (pkt.vector_data[i] != 0) {
                data_used = i + 1;
                break;
            }
        }
        result.data_used_byte = HEADER_BYTE + BALL_DATA_BYTE + data_used;
        result.packet = pkt;
        return result;
    }
};

// ===================== 路径配置 =====================
const string VIDEO_FILE_PATH = "/home/chenjie/proo/search-cpp/test_video1.avi";
const string OUTPUT_VIDEO_PATH = "/home/chenjie/proo/search-cpp/output_final_fix.avi";
const string OUTPUT_FRAMES_DIR = "/home/chenjie/proo/search-cpp/output_frames_final_fix/";

bool createDir(const string& path) { return system(("mkdir -p " + path).c_str()) == 0; }

// ===================== 主函数 =====================
int main() {
    cout << "========================================" << endl;
    cout << "  最终修正版：解决类型不匹配编译错误" << endl;
    cout << "========================================" << endl;

    if (!createDir(OUTPUT_FRAMES_DIR)) { cerr << "[错误] 无法创建目录" << endl; return -1; }

    VideoCapture cap(VIDEO_FILE_PATH);
    if (!cap.isOpened()) { cerr << "[错误] 无法打开视频" << endl; return -1; }

    int origWidth = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int origHeight = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double origFps = cap.get(CAP_PROP_FPS);
    int origTotalFrames = (int)cap.get(CAP_PROP_FRAME_COUNT);
    int origFourcc = (int)cap.get(CAP_PROP_FOURCC);
    double origDuration = origTotalFrames / origFps;

    int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');
    VideoWriter writer(OUTPUT_VIDEO_PATH, fourcc, origFps, Size(origWidth * 2, origHeight), true);
    if (!writer.isOpened()) { cerr << "[错误] 无法创建输出视频" << endl; return -1; }

    HeroCamCompressor compressor;
    Mat frame;
    uint8_t frame_seq = 0;
    int processedFrames = 0;

    vector<int> frame_sizes;
    vector<int> extracted_lines;
    vector<int> stored_lines;
    vector<int> ball_counts;
    auto start_time = high_resolution_clock::now();

    cout << "\n[逐帧处理中...（完整处理所有帧）]" << endl;
    cout << "----------------------------------------" << endl;
    cout << "  帧号  |  总帧数  |  识别弹丸  |  存储线段  |  压缩大小" << endl;
    cout << "----------------------------------------" << endl;

    while (true) {
        if (!cap.read(frame)) {
            cout << "\n[提示] 视频已处理完毕，到达最后一帧！" << endl;
            break;
        }
        processedFrames++;
        frame_seq++;

        ProcessResult result = compressor.process(frame);
        result.packet.frame_seq = frame_seq;

        frame_sizes.push_back(result.data_used_byte);
        extracted_lines.push_back(result.lines.size());
        stored_lines.push_back(result.line_count);
        ball_counts.push_back(result.ballCount);

        cout << "  " << setw(4) << processedFrames 
             << "  |  " << setw(4) << origTotalFrames
             << "  |  " << setw(6) << result.ballCount
             << "  |  " << setw(6) << result.line_count
             << "  |  " << setw(4) << result.data_used_byte << " B" << endl;

        // ========== 终端机解码重绘 ==========
        MqttPacket received_packet = result.packet;
        Mat decoded_display = Mat::zeros(origHeight, origWidth, CV_8UC3);
        bool is_png_mode = (received_packet.config & CONFIG_MODE_VECTOR) != 0;

        if (!is_png_mode) {
            vector<Vec4i> decoded_lines;
            for (int i = 0; i < VECTOR_DATA_MAX_BYTE / LINE_PAIR_BYTE_SIZE; i++) {
                int offset = i * LINE_PAIR_BYTE_SIZE;
                Vec4i l1, l2;
                unpackLinePair(received_packet.vector_data + offset, l1, l2);
                if (l1[0] != 0 || l1[1] != 0 || l1[2] != 0 || l1[3] != 0) decoded_lines.push_back(l1);
                if (l2[0] != 0 || l2[1] != 0 || l2[2] != 0 || l2[3] != 0) decoded_lines.push_back(l2);
                if (decoded_lines.size() >= MAX_LINE_COUNT) break;
            }

            for (const auto& l : decoded_lines) {
                cv::line(decoded_display, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 2, LINE_AA);
            }

            Point ball1, ball2;
            int r1, r2;
            bool valid1, valid2;
            unpackBall(received_packet.ball_data, ball1, r1, valid1);
            unpackBall(received_packet.ball_data + 4, ball2, r2, valid2);
            
            if (valid1) {
                cv::circle(decoded_display, ball1, r1, Scalar(255, 255, 255), -1, LINE_AA);
                cv::circle(decoded_display, ball1, r1 + 3, Scalar(0, 255, 0), 3, LINE_AA);
            }
            if (valid2) {
                cv::circle(decoded_display, ball2, r2, Scalar(255, 255, 255), -1, LINE_AA);
                cv::circle(decoded_display, ball2, r2 + 3, Scalar(0, 255, 0), 3, LINE_AA);
            }
        }

        Mat final_output = Mat::zeros(origHeight, origWidth * 2, CV_8UC3);
        result.originalMarked.copyTo(final_output(Rect(0, 0, origWidth, origHeight)));
        decoded_display.copyTo(final_output(Rect(origWidth, 0, origWidth, origHeight)));
        putText(final_output, "Original (All)", Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        putText(final_output, "Decoded (Final Fix)", Point(origWidth + 20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);

        char frame_path[256];
        sprintf(frame_path, "%s/frame_%06d.png", OUTPUT_FRAMES_DIR.c_str(), processedFrames);
        imwrite(frame_path, final_output);
        writer.write(final_output);

        imshow("Final Fix", final_output);
        waitKey(1);
    }

    auto end_time = high_resolution_clock::now();
    double total_time = duration_cast<duration<double>>(end_time - start_time).count();

    cap.release();
    writer.release();
    destroyAllWindows();

    // 计算统计数据
    int min_size = *min_element(frame_sizes.begin(), frame_sizes.end());
    int max_size = *max_element(frame_sizes.begin(), frame_sizes.end());
    double avg_size = accumulate(frame_sizes.begin(), frame_sizes.end(), 0.0) / frame_sizes.size();
    long long total_size = accumulate(frame_sizes.begin(), frame_sizes.end(), 0LL);

    double avg_extracted_lines = accumulate(extracted_lines.begin(), extracted_lines.end(), 0.0) / extracted_lines.size();
    double avg_stored_lines = accumulate(stored_lines.begin(), stored_lines.end(), 0.0) / stored_lines.size();
    double avg_balls = accumulate(ball_counts.begin(), ball_counts.end(), 0.0) / ball_counts.size();
    int max_balls = *max_element(ball_counts.begin(), ball_counts.end());

    int outputWidth = origWidth * 2;
    int outputHeight = origHeight;
    double outputFps = origFps;

    // 最终统计报告
    cout << "\n\n" << string(80, '=') << endl;
    cout << "                    最终统计报告" << endl;
    cout << string(80, '=') << endl;

    cout << "\n【1. 原视频参数】" << endl;
    cout << "----------------------------------------" << endl;
    cout << "  文件路径：        " << VIDEO_FILE_PATH << endl;
    cout << "  分辨率：          " << origWidth << " × " << origHeight << endl;
    cout << "  总帧数：          " << origTotalFrames << " 帧" << endl;
    cout << "  实际处理帧数：    " << processedFrames << " 帧" << endl;
    cout << "  帧率：            " << fixed << setprecision(2) << origFps << " FPS" << endl;
    cout << "  时长：            " << fixed << setprecision(2) << origDuration << " 秒" << endl;

    cout << "\n【2. 压缩数据统计】" << endl;
    cout << "----------------------------------------" << endl;
    cout << "  数据包固定大小：  " << TOTAL_PACKET_BYTE << " 字节" << endl;
    cout << "  压缩数据最小：    " << min_size << " 字节" << endl;
    cout << "  压缩数据最大：    " << max_size << " 字节" << endl;
    cout << "  压缩数据平均：    " << fixed << setprecision(2) << avg_size << " 字节" << endl;
    cout << "  总传输数据量：    " << fixed << setprecision(2) << total_size / 1024.0 << " KB" << endl;

    cout << "\n【3. 弹丸统计】" << endl;
    cout << "----------------------------------------" << endl;
    cout << "  单帧最多弹丸：    " << max_balls << " 个" << endl;
    cout << "  单帧平均弹丸：    " << fixed << setprecision(1) << avg_balls << " 个" << endl;

    cout << "\n【4. 输出视频参数】" << endl;
    cout << "----------------------------------------" << endl;
    cout << "  文件路径：        " << OUTPUT_VIDEO_PATH << endl;
    cout << "  分辨率：          " << outputWidth << " × " << outputHeight << " (双画面)" << endl;
    cout << "  帧率：            " << fixed << setprecision(2) << outputFps << " FPS" << endl;
    cout << "  总处理时间：      " << fixed << setprecision(2) << total_time << " 秒" << endl;

    cout << "\n" << string(80, '=') << endl;
    cout << "  处理完成！输出视频：" << OUTPUT_VIDEO_PATH << endl;
    cout << string(80, '=') << endl;

    return 0;
}
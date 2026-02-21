#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cfloat>

#undef min
#undef max

using namespace cv;
using namespace std;
using namespace std::chrono;

// ==============================================
// 核心硬约束
// ==============================================
const int TOTAL_PACKET_BYTE = 300;
const int HEADER_BYTE = 6;
const int BALL_DATA_BYTE = 8;
const int VECTOR_DATA_MAX_BYTE = TOTAL_PACKET_BYTE - HEADER_BYTE - BALL_DATA_BYTE;
static_assert(HEADER_BYTE + BALL_DATA_BYTE + VECTOR_DATA_MAX_BYTE == TOTAL_PACKET_BYTE, "300字节硬约束校验失败");

const Size ORIGINAL_SIZE = Size(1350, 1080);
const int COORD_BITS = 9;
const int COORD_MAX = (1 << COORD_BITS) - 1;
const double SCALE_FACTOR = (double)COORD_MAX / std::max(ORIGINAL_SIZE.width, ORIGINAL_SIZE.height);
const int LINE_PAIR_BYTE_SIZE = 9;
const int MAX_LINE_COUNT = 71;
const int BALL_TRAJECTORY_ROI = 150;

#define CONFIG_FRAME_VALID     0x01
#define CONFIG_MODE_VECTOR     0x02
#define CONFIG_BALL_VALID      0x04

// ===================== 识别参数 =====================
const float MIN_BALL_AREA = 100.0f;          // 保持不变，过滤小噪点
const float MAX_BALL_AREA = 8000.0f;         // 【修改】从2000提升到8000，识别更大的弹丸
const float MIN_BALL_CIRCULARITY = 0.7f;      // 【修改】从0.8微降到0.7，增加容错
const float MAX_BALL_ASPECT_RATIO = 1.5f;     // 【修改】从1.3微升到1.5，增加容错
// 【核心修改】HSV范围大幅放宽，识别更暗的绿色
// H: 25-100 (覆盖更广的绿色范围), S: 3-255 (允许低饱和度暗绿), V: 40-255 (允许很暗的绿色)
const Scalar BALL_HSV_LOW = Scalar(35, 3, 40);
const Scalar BALL_HSV_HIGH = Scalar(85, 255, 255);

const float HOUGH_RHO = 1.0f;
const float HOUGH_THETA = CV_PI / 180.0f;
const int HOUGH_THRESHOLD = 15;
const double HOUGH_MIN_LINE_LENGTH = 5.0;
const double HOUGH_MAX_LINE_GAP = 10.0;

// 保守去重参数
const double MERGE_ANGLE_THRESHOLD = 2.0;
const double MERGE_DISTANCE_THRESHOLD = 8.0;

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

struct ProcessResult {
    cv::Mat originalMarked;
    MqttPacket packet;
    int line_count;
    int ballCount;
    int raw_line_count;
    int merged_line_count;
    bool did_merge;
    vector<Vec4i> lines;
    vector<Point2f> ballCenters;
    vector<float> ballRadii;
    vector<vector<Point>> ballContours; // 新增：保存弹丸原始轮廓
    int data_used_byte;
};

// ===================== 工具函数 =====================
inline uint16_t quantizeCoord(int val) {
    return (uint16_t)round(val * SCALE_FACTOR);
}
inline int dequantizeCoord(uint16_t val) {
    return (int)round(val / SCALE_FACTOR);
}

inline void packBall(uint8_t* buf, Point center, int radius, bool valid) {
    buf[0] = buf[1] = buf[2] = buf[3] = 0;
    if (!valid) return;
    uint16_t x = quantizeCoord(center.x);
    uint16_t y = quantizeCoord(center.y);
    uint8_t r = (uint8_t)std::min(radius, 63);
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

inline bool isLineInBallROI(Vec4i line, const vector<Point2f>& ballCenters, int roi_range) {
    if (ballCenters.empty()) return false;
    Point2f p1((float)line[0], (float)line[1]);
    Point2f p2((float)line[2], (float)line[3]);
    for (const auto& center : ballCenters) {
        if (cv::norm(p1 - center) < roi_range || cv::norm(p2 - center) < roi_range) {
            return true;
        }
    }
    return false;
}

inline double getLineAngle(const Vec4i& line) {
    double dx = line[2] - line[0];
    double dy = line[3] - line[1];
    double angle = atan2(dy, dx) * 180.0 / CV_PI;
    if (angle < 0) angle += 180.0;
    return angle;
}

inline double getLineLength(const Vec4i& line) {
    return cv::norm(Point(line[0], line[1]) - Point(line[2], line[3]));
}

inline double pointToLineDistance(const Point2f& p, const Vec4i& line) {
    Point2f p1(line[0], line[1]);
    Point2f p2(line[2], line[3]);
    double nom = abs((p2.y - p1.y)*p.x - (p2.x - p1.x)*p.y + p2.x*p1.y - p2.y*p1.x);
    double den = cv::norm(p2 - p1);
    return den < 1e-6 ? cv::norm(p - p1) : nom / den;
}

// ===================== 保守线段去重逻辑 =====================
vector<Vec4i> conservativeLineDeduplication(const vector<Vec4i>& lines) {
    if (lines.size() <= MAX_LINE_COUNT) {
        return lines;
    }
    if (lines.empty()) return {};

    vector<pair<double, size_t>> length_idx;
    length_idx.reserve(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        length_idx.emplace_back(-getLineLength(lines[i]), i);
    }
    sort(length_idx.begin(), length_idx.end());

    vector<Vec4i> result;
    vector<bool> consumed(lines.size(), false);
    result.reserve(MAX_LINE_COUNT);

    for (size_t i = 0; i < length_idx.size() && result.size() < MAX_LINE_COUNT; ++i) {
        size_t main_idx = length_idx[i].second;
        if (consumed[main_idx]) continue;

        const Vec4i& main_line = lines[main_idx];
        const double main_len = getLineLength(main_line);
        const double main_angle = getLineAngle(main_line);
        
        result.push_back(main_line);
        consumed[main_idx] = true;

        for (size_t j = i + 1; j < length_idx.size(); ++j) {
            size_t curr_idx = length_idx[j].second;
            if (consumed[curr_idx]) continue;

            const Vec4i& curr_line = lines[curr_idx];
            double curr_angle = getLineAngle(curr_line);
            double angle_diff = abs(main_angle - curr_angle);
            angle_diff = angle_diff > 90 ? 180 - angle_diff : angle_diff;
            if (angle_diff > MERGE_ANGLE_THRESHOLD) continue;

            Point2f c1(curr_line[0], curr_line[1]);
            Point2f c2(curr_line[2], curr_line[3]);
            double d1 = pointToLineDistance(c1, main_line);
            double d2 = pointToLineDistance(c2, main_line);
            if (d1 > MERGE_DISTANCE_THRESHOLD || d2 > MERGE_DISTANCE_THRESHOLD) continue;

            double curr_len = getLineLength(curr_line);
            if (curr_len > main_len * 0.8) continue;

            consumed[curr_idx] = true;
        }
    }

    return result;
}

// ===================== 核心处理器类 =====================
class HeroCamCompressor
{
public:
    ProcessResult process(Mat &input)
    {
        ProcessResult result;
        if (input.empty()) return result;
        const int origW = input.cols;
        const int origH = input.rows;

        // 1. 边缘提取
        Mat gray, blurred, edges;
        cvtColor(input, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurred, Size(5, 5), 1.3);
        Canny(blurred, edges, 50, 150);
        
        // 2. 霍夫直线变换
        vector<Vec4i> all_lines;
        HoughLinesP(edges, all_lines, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);
        result.raw_line_count = all_lines.size();
        result.did_merge = false;

        // 3. 智能线段处理
        vector<Vec4i> processed_lines;
        if (all_lines.size() <= MAX_LINE_COUNT) {
            processed_lines = all_lines;
            result.merged_line_count = all_lines.size();
        } else {
            processed_lines = conservativeLineDeduplication(all_lines);
            result.merged_line_count = processed_lines.size();
            result.did_merge = true;
        }

        // 4. 弹丸识别
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

        for (const auto &cnt : ballContours) {
            const double area = contourArea(cnt);
            if (area < MIN_BALL_AREA || area > MAX_BALL_AREA) continue;
            const double perim = arcLength(cnt, true);
            if (perim <= 0) continue;
            const double circularity = 4.0 * CV_PI * area / (perim * perim);
            if (circularity < MIN_BALL_CIRCULARITY) continue;
            const Rect rect = boundingRect(cnt);
            double aspect = static_cast<double>(rect.width) / rect.height;
            aspect = aspect < 1.0 ? 1.0 / aspect : aspect;
            if (aspect > MAX_BALL_ASPECT_RATIO) continue;

            Point2f center;
            float radius;
            minEnclosingCircle(cnt, center, radius);
            result.ballCenters.push_back(center);
            result.ballRadii.push_back(radius);
            result.ballContours.push_back(cnt); // 保存原始轮廓

            if (!valid1) { ball1 = Point(cvRound(center.x), cvRound(center.y)); r1 = cvRound(radius); valid1 = true; }
            else if (!valid2) { ball2 = Point(cvRound(center.x), cvRound(center.y)); r2 = cvRound(radius); valid2 = true; }
            validBalls++;
        }
        result.ballCount = validBalls;

        // 5. 线段优先级筛选
        vector<Vec4i> trajectory_lines;
        vector<Vec4i> field_lines;
        
        for (const auto& line : processed_lines) {
            if (isLineInBallROI(line, result.ballCenters, BALL_TRAJECTORY_ROI)) {
                trajectory_lines.push_back(line);
            } else {
                field_lines.push_back(line);
            }
        }

        sort(field_lines.begin(), field_lines.end(), [](const Vec4i& a, const Vec4i& b) {
            return getLineLength(a) > getLineLength(b);
        });

        vector<Vec4i> final_lines;
        final_lines.reserve(MAX_LINE_COUNT);
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

        // ===================== 【核心修改】可视化绘制 =====================
        // 首先把原始画面的内容复制过来（保留弹丸原始颜色）
        Mat originalMarked;
        input.copyTo(originalMarked);
        
        // 绘制线段
        const auto& lines_to_draw = result.did_merge ? processed_lines : all_lines;
        for (const auto& l : lines_to_draw) {
            cv::line(originalMarked, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(200, 200, 200), 1, LINE_AA);
        }
        for (const auto& l : final_lines) {
            cv::line(originalMarked, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 2, LINE_AA);
        }
        
        // 【核心修改】绘制弹丸：保留原始形状 + 绿色空心外接圆
        for (size_t i = 0; i < result.ballContours.size(); i++) {
            // 1. 绘制绿色原始轮廓（保留弹丸原始形状）
            drawContours(originalMarked, result.ballContours, (int)i, Scalar(0, 255, 0), 2, LINE_AA);
            
            // 2. 绘制绿色空心最小外接圆（套在外面）
            cv::circle(originalMarked, result.ballCenters[i], (int)result.ballRadii[i], Scalar(0, 255, 0), 2, LINE_AA);
        }
        
        string status_text = result.did_merge ? "Status: Deduplicated" : "Status: Raw (No Merge)";
        putText(originalMarked, status_text, Point(20, origH - 20), FONT_HERSHEY_SIMPLEX, 0.8, result.did_merge ? Scalar(0, 165, 255) : Scalar(0, 255, 0), 2);
        result.originalMarked = originalMarked;

        // 6. 打包
        MqttPacket pkt;
        memset(&pkt, 0, sizeof(MqttPacket));
        pkt.config = CONFIG_FRAME_VALID;
        pkt.width = origW;
        pkt.height = origH;

        packBall(pkt.ball_data, ball1, r1, valid1);
        packBall(pkt.ball_data + 4, ball2, r2, valid2);
        if (valid1 || valid2) pkt.config |= CONFIG_BALL_VALID;

        memset(pkt.vector_data, 0, VECTOR_DATA_MAX_BYTE);
        const int line_count = final_lines.size();
        const int pair_count = line_count / 2;
        const int single_line = line_count % 2;
        
        for (int i = 0; i < pair_count; i++) {
            const int offset = i * LINE_PAIR_BYTE_SIZE;
            if (offset + LINE_PAIR_BYTE_SIZE > VECTOR_DATA_MAX_BYTE) break;
            packLinePair(pkt.vector_data + offset, final_lines[i*2], final_lines[i*2+1]);
        }
        
        if (single_line && (pair_count * LINE_PAIR_BYTE_SIZE + 5) <= VECTOR_DATA_MAX_BYTE) {
            const Vec4i dummy_line(0,0,0,0);
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
const string VIDEO_FILE_PATH = "/home/chenjie/proo/search-cpp/_2025-12-14_19_50_07_151.avi";
const string OUTPUT_VIDEO_PATH = "/home/chenjie/proo/search-cpp/output_final_fix.avi";
const string OUTPUT_FRAMES_DIR = "/home/chenjie/proo/search-cpp/output_frames_final_fix/";

bool createDir(const string& path) { return system(("mkdir -p " + path).c_str()) == 0; }

// ===================== 主函数 =====================
int main() {
    cout << "========================================" << endl;
    cout << "  弹丸可视化优化版：保留原始形状 + 绿色外接圆" << endl;
    cout << "========================================" << endl;

    if (!createDir(OUTPUT_FRAMES_DIR)) { cerr << "[错误] 无法创建输出目录" << endl; return -1; }

    VideoCapture cap(VIDEO_FILE_PATH);
    if (!cap.isOpened()) { cerr << "[错误] 无法打开输入视频" << endl; return -1; }

    // 获取原始视频信息
    const int origWidth = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    const int origHeight = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    const double origFps = cap.get(CAP_PROP_FPS);
    const int origTotalFrames = (int)cap.get(CAP_PROP_FRAME_COUNT);
    const double origDuration = origTotalFrames / origFps;
    
    cout << "\n【原始视频信息】" << endl;
    cout << "----------------------------------------" << endl;
    cout << "  文件路径:    " << VIDEO_FILE_PATH << endl;
    cout << "  分辨率:      " << origWidth << " × " << origHeight << endl;
    cout << "  帧率:        " << fixed << setprecision(2) << origFps << " FPS" << endl;
    cout << "  总帧数:      " << origTotalFrames << endl;
    cout << "  视频时长:    " << fixed << setprecision(2) << origDuration << " 秒" << endl;
    cout << "----------------------------------------" << endl;

    const int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');
    VideoWriter writer(OUTPUT_VIDEO_PATH, fourcc, origFps, Size(origWidth * 2, origHeight), true);
    if (!writer.isOpened()) { cerr << "[错误] 无法创建输出视频" << endl; return -1; }

    HeroCamCompressor compressor;
    Mat frame;
    uint8_t frame_seq = 0;
    int processedFrames = 0;
    int frames_merged = 0;

    // 统计数据收集
    vector<int> frame_sizes;
    vector<int> raw_line_counts;
    vector<int> merged_line_counts;
    vector<int> stored_line_counts;
    vector<int> ball_counts;
    
    auto start_time = high_resolution_clock::now();
    
    cout << "\n【开始处理...】" << endl;
    cout << "  帧号  |  霍夫线  |  处理后  |  存储线  |  弹丸  |  包大小" << endl;
    cout << "----------------------------------------" << endl;

    while (true) {
        if (!cap.read(frame)) break;
        processedFrames++;
        frame_seq++;

        ProcessResult result = compressor.process(frame);
        result.packet.frame_seq = frame_seq;

        if (result.did_merge) frames_merged++;
        
        // 收集统计数据
        frame_sizes.push_back(result.data_used_byte);
        raw_line_counts.push_back(result.raw_line_count);
        merged_line_counts.push_back(result.merged_line_count);
        stored_line_counts.push_back(result.line_count);
        ball_counts.push_back(result.ballCount);

        cout << "  " << setw(4) << processedFrames 
             << "  |  " << setw(6) << result.raw_line_count
             << "  |  " << setw(6) << result.merged_line_count
             << "  |  " << setw(6) << result.line_count
             << "  |  " << setw(4) << result.ballCount
             << "  |  " << setw(4) << result.data_used_byte << " B" << endl;

        // 解码重绘（右侧画面）
        const MqttPacket& received_packet = result.packet;
        Mat decoded_display = Mat::zeros(origHeight, origWidth, CV_8UC3);
        const bool is_png_mode = (received_packet.config & CONFIG_MODE_VECTOR) != 0;

        if (!is_png_mode) {
            vector<Vec4i> decoded_lines;
            for (int i = 0; i < VECTOR_DATA_MAX_BYTE / LINE_PAIR_BYTE_SIZE; i++) {
                const int offset = i * LINE_PAIR_BYTE_SIZE;
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
            
            // 右侧解码画面也保持一致：绿色空心圆
            if (valid1) {
                cv::circle(decoded_display, ball1, r1, Scalar(0, 255, 0), 2, LINE_AA);
            }
            if (valid2) {
                cv::circle(decoded_display, ball2, r2, Scalar(0, 255, 0), 2, LINE_AA);
            }
        }

        Mat final_output = Mat::zeros(origHeight, origWidth * 2, CV_8UC3);
        result.originalMarked.copyTo(final_output(Rect(0, 0, origWidth, origHeight)));
        decoded_display.copyTo(final_output(Rect(origWidth, 0, origWidth, origHeight)));
        
        line(final_output, Point(origWidth, 0), Point(origWidth, origHeight), Scalar(0,0,255), 2);
        putText(final_output, "Input (Left)", Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        putText(final_output, "Output (Right)", Point(origWidth + 20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);

        char frame_path[256];
        sprintf(frame_path, "%s/frame_%06d.png", OUTPUT_FRAMES_DIR.c_str(), processedFrames);
        imwrite(frame_path, final_output);
        writer.write(final_output);

        imshow("Stable Processing", final_output);
        if (waitKey(1) == 27) break;
    }

    auto end_time = high_resolution_clock::now();
    double total_time = duration_cast<duration<double>>(end_time - start_time).count();
    cap.release();
    writer.release();
    destroyAllWindows();

    // 详细统计报告
    if (processedFrames > 0) {
        int min_size = *min_element(frame_sizes.begin(), frame_sizes.end());
        int max_size = *max_element(frame_sizes.begin(), frame_sizes.end());
        double avg_size = accumulate(frame_sizes.begin(), frame_sizes.end(), 0.0) / frame_sizes.size();
        
        int min_raw_line = *min_element(raw_line_counts.begin(), raw_line_counts.end());
        int max_raw_line = *max_element(raw_line_counts.begin(), raw_line_counts.end());
        double avg_raw_line = accumulate(raw_line_counts.begin(), raw_line_counts.end(), 0.0) / raw_line_counts.size();
        
        int min_merged_line = *min_element(merged_line_counts.begin(), merged_line_counts.end());
        int max_merged_line = *max_element(merged_line_counts.begin(), merged_line_counts.end());
        double avg_merged_line = accumulate(merged_line_counts.begin(), merged_line_counts.end(), 0.0) / merged_line_counts.size();
        
        int min_stored_line = *min_element(stored_line_counts.begin(), stored_line_counts.end());
        int max_stored_line = *max_element(stored_line_counts.begin(), stored_line_counts.end());
        double avg_stored_line = accumulate(stored_line_counts.begin(), stored_line_counts.end(), 0.0) / stored_line_counts.size();
        
        int min_balls = *min_element(ball_counts.begin(), ball_counts.end());
        int max_balls = *max_element(ball_counts.begin(), ball_counts.end());
        double avg_balls = accumulate(ball_counts.begin(), ball_counts.end(), 0.0) / ball_counts.size();

        cout << "\n\n";
        cout << "╔══════════════════════════════════════════════════════════════╗" << endl;
        cout << "║                    最终处理统计报告                            ║" << endl;
        cout << "╠══════════════════════════════════════════════════════════════╣" << endl;
        cout << "║  【原始视频信息】                                              ║" << endl;
        cout << "║  文件路径:    " << left << setw(45) << VIDEO_FILE_PATH << "║" << endl;
        cout << "║  分辨率:      " << left << setw(10) << origWidth << " × " << left << setw(10) << origHeight << "          ║" << endl;
        cout << "║  帧率:        " << left << fixed << setprecision(2) << setw(15) << origFps << " FPS                ║" << endl;
        cout << "║  总帧数:      " << left << setw(15) << origTotalFrames << " 帧                  ║" << endl;
        cout << "║  视频时长:    " << left << fixed << setprecision(2) << setw(15) << origDuration << " 秒                 ║" << endl;
        cout << "╠══════════════════════════════════════════════════════════════╣" << endl;
        cout << "║  【处理过程统计】                                              ║" << endl;
        cout << "║  处理总帧数:  " << left << setw(15) << processedFrames << " 帧                  ║" << endl;
        cout << "║  总处理耗时:  " << left << fixed << setprecision(2) << setw(15) << total_time << " 秒                 ║" << endl;
        cout << "║  平均处理速度:" << left << fixed << setprecision(1) << setw(15) << processedFrames / total_time << " FPS                ║" << endl;
        cout << "║  进行去重:    " << left << setw(15) << frames_merged << " 帧                  ║" << endl;
        cout << "║  保持原始:    " << left << setw(15) << (processedFrames - frames_merged) << " 帧                  ║" << endl;
        cout << "╠══════════════════════════════════════════════════════════════╣" << endl;
        cout << "║  【线段统计】                最小值    最大值    平均值       ║" << endl;
        cout << "║  霍夫原始线段:    " << right << setw(8) << min_raw_line << "   " << right << setw(8) << max_raw_line << "   " << right << fixed << setprecision(1) << setw(8) << avg_raw_line << "       ║" << endl;
        cout << "║  去重处理后:      " << right << setw(8) << min_merged_line << "   " << right << setw(8) << max_merged_line << "   " << right << fixed << setprecision(1) << setw(8) << avg_merged_line << "       ║" << endl;
        cout << "║  最终存储线段:    " << right << setw(8) << min_stored_line << "   " << right << setw(8) << max_stored_line << "   " << right << fixed << setprecision(1) << setw(8) << avg_stored_line << "       ║" << endl;
        cout << "║  存储上限:        " << right << setw(8) << MAX_LINE_COUNT << "   " << right << setw(8) << MAX_LINE_COUNT << "   " << right << setw(8) << MAX_LINE_COUNT << "       ║" << endl;
        cout << "╠══════════════════════════════════════════════════════════════╣" << endl;
        cout << "║  【弹丸统计】                最小值    最大值    平均值       ║" << endl;
        cout << "║  识别弹丸数:      " << right << setw(8) << min_balls << "   " << right << setw(8) << max_balls << "   " << right << fixed << setprecision(1) << setw(8) << avg_balls << "       ║" << endl;
        cout << "╠══════════════════════════════════════════════════════════════╣" << endl;
        cout << "║  【数据包统计】              最小值    最大值    平均值       ║" << endl;
        cout << "║  数据包大小:      " << right << setw(8) << min_size << "   " << right << setw(8) << max_size << "   " << right << fixed << setprecision(1) << setw(8) << avg_size << "       ║" << endl;
        cout << "║  数据包上限:      " << right << setw(8) << TOTAL_PACKET_BYTE << "   " << right << setw(8) << TOTAL_PACKET_BYTE << "   " << right << setw(8) << TOTAL_PACKET_BYTE << "       ║" << endl;
        cout << "╠══════════════════════════════════════════════════════════════╣" << endl;
        cout << "║  【输出视频信息】                                              ║" << endl;
        cout << "║  输出视频:    " << left << setw(45) << OUTPUT_VIDEO_PATH << "║" << endl;
        cout << "║  输出帧图:    " << left << setw(45) << OUTPUT_FRAMES_DIR << "║" << endl;
        cout << "║  输出分辨率:  " << left << setw(10) << origWidth * 2 << " × " << left << setw(10) << origHeight << "          ║" << endl;
        cout << "║  输出帧率:    " << left << fixed << setprecision(2) << setw(15) << origFps << " FPS                ║" << endl;
        cout << "╚══════════════════════════════════════════════════════════════╝" << endl;
    }

    return 0;
}
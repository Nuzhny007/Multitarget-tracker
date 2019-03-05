

#include <opencv2/opencv.hpp>
#include <armadillo>
#include "global.hpp"
#include "lmbTracker.hpp"
#include "simulate.hpp"

int main() {
    lmbTracker filter;

#if 0
    Simulate sim;
    std::vector<arma::fvec> Zk, true_Zk;
    std::vector<std::pair<lmbTracker::label_t, arma::fvec>> Xk;
    for (int i=1; i<=100; i++) {
        printf("Iteration:%d\r\n", i);
        sim.runSimulation();
        Zk = sim.getSensorMeasurements();
        true_Zk = sim.getTrueObservations();
        for (auto &zk : true_Zk) {
            zk.print("zk");
        }
        //        for (auto &zk : Zk) {
        //      zk.print("zk");
        //    }
        std::cout << "--------" << std::endl;
        Xk = filter.runFilter(Zk);
        for (auto &x : Xk) {
            std::cout << "<" << x.first.first << ","
                      << x.first.second << ">" << std::endl;
            x.second.print();
        }
    }
#endif

#if 1
    std::map<int, std::vector<cv::Rect>> detections;
    // getting in the detections
    std::ifstream data("../../dataset_videos/KITTI-17/det/det.txt");
    // std::ifstream data("../../dataset_videos/PETS09-S2L1/det/det.txt");
    if (!data.is_open()) {
        std::cout << "couldn't read detections file" << std::endl;
        return -1;
    }
    std::string line;
    while (std::getline(data, line)) {
        int frame;
        float x, y, w, h;

        std::stringstream lineStream(line);
        std::string cell;
        int n_cell = 0;
        while (std::getline(lineStream, cell, ',')) {
            n_cell++;
            if (n_cell == 7) {
                break;
            }

            switch (n_cell) {
            case 1:
                frame = atoi(cell.c_str());
                break;
            case 3:
                x = strtod(cell.c_str(), NULL);
                break;
            case 4:
                y = strtod(cell.c_str(), NULL);
                break;
            case 5:
                w = strtod(cell.c_str(), NULL);
                break;
            case 6:
                h = strtod(cell.c_str(), NULL);
                detections[frame].emplace_back(x, y, w, h);
                break;
            }
        }
    }

    std::vector<arma::fvec> Zk;
    std::vector<std::pair<std::pair<unsigned, unsigned>, arma::fvec>> Xk;
    cv::VideoCapture vc("../../dataset_videos/KITTI-17/img1/%6d.jpg");
    // cv::VideoCapture vc("../../dataset_videos/PETS09-S2L1/img1/%6d.jpg");
    if (!vc.isOpened()) {
        std::cout << "image sequence not read" << std::endl;
        return -1;
    }
    cv::Mat frame;
    vc >> frame;
    cv::VideoWriter vw;
    vw.open("out.avi", cv::VideoWriter::fourcc('x', 'v', 'i', 'd'), 10, frame.size(), true);
    if (!vw.isOpened()) {
        std::cout << "cant create video writer" << std::endl;
        return -1;
    }
    unsigned frame_cnt = 1;

    float sum_fps = 0;
    float worst_case_fps = INF;
    double WC_frame_time = 0.0;
    unsigned WC_iteration;
    unsigned iteration = 0;
    while (!frame.empty()) {
        if (frame_cnt % 2 == 0) {
            vc >> frame;
            frame_cnt++;
            // only processing alternate frames so as to have smoothness in input
            // and filter is not influenced by rapid/abrupt transitions in inputs
            continue;
        }
        iteration++;
        Zk.clear();
        //    std::cout << "Iteration:" << frame_cnt << std::endl << "-------------"
        //    << std::endl;
        for (auto &d : detections[frame_cnt]) {
            Zk.emplace_back(arma::fvec(
            {(d.x + d.width / (float)2), (d.y + d.height / (float)2)}));
            //      (Zk.end() - 1)->print("meas");

            //            cv::Rect rect;
            //            rect.height = 10;
            //            rect.width = 10;
            //            rect.x = std::max(0.0, (*(Zk.end()-1))(0) - rect.width/2.0);
            //            rect.y = std::max(0.0, (*(Zk.end()-1))(1) -
            //            rect.height/2.0);
            //            cv::rectangle(frame, rect.tl(), rect.br(),
            //            cv::Scalar(0,0,255), 3);
        }
        int64 start = cv::getTickCount();
        Xk = filter.runFilter(Zk);
        double time_taken =
                (double)(cv::getTickCount() - start) / (double)cv::getTickFrequency();
        if (WC_frame_time < time_taken) {
            WC_frame_time = time_taken;
            WC_iteration = iteration;
        }
        //    float this_fps = (cv::getTickFrequency() / (cv::getTickCount() -
        //    start));
        float this_fps =
                (cv::getTickFrequency() / (cv::getTickCount() - start)) * 2;
        if (frame_cnt != 1) {
            sum_fps += this_fps;
            if (worst_case_fps > this_fps) {
                worst_case_fps = this_fps;
            }
        }
        cv::putText(frame, "lmb:" + std::to_string(this_fps) + "fps",
                    cv::Point(5, 25), cv::FONT_HERSHEY_SIMPLEX, 1.,
                    cv::Scalar(255, 255, 255), 2);

        for (auto &x : Xk) {
            //      std::cout << "<" << x.first.first << "," << x.first.second << ">"
            //      <<
            //      std::endl;
            //      x.second.print();
            cv::Rect rect;
            rect.height = 10;
            rect.width = 10;
            rect.x = std::max(0.0, x.second(0) - rect.width / 2.0);
            rect.y = std::max(0.0, x.second(2) - rect.height / 2.0);
            //      cv::putText(
            //                frame,
            //                "<" + std::to_string(x.first.first) + ","
            //                        + std::to_string(x.first.second) + ">",
            //                cv::Point(x.second(0), x.second(2)),
            //                cv::FONT_HERSHEY_SIMPLEX,
            //                0.4,
            //                cv::Scalar(0,255,0),
            //                2
            //            );
            cv::rectangle(frame, rect.tl(), rect.br(), cv::Scalar(0, 255, 0), 3);
        }
        //    std::cout << "-------------------------------------" << std::endl;

        cv::imshow("tracked_video", frame);

        if (cv::waitKey(30) >= 0) {
            break;
        }
        //    cv::waitKey(0);

        vw << frame;
        vc >> frame;
        frame_cnt++;
    }

    vc.release();
    vw.release();

    //    std::cout << "AVG_FPS: " << sum_fps/frame_cnt << std::endl;
    //  std::cout << "WC_FPS: " << worst_case_fps << " for frame: " << std::endl;
    std::cout << "WC_frame_time:" << WC_frame_time * 1000
              << "for iteration:" << WC_iteration << std::endl;
    std::cout << std::endl;
#endif
}

// TODO:
// b) robustness (like unknown clutter profile, unknown detection probabilities
// etc)
// c) phd lookahead to reduce k-shortestpath and rank assignment problem
// computations
// d) grouping and gating of lmb predict components to enable efficient
// parallelized
//    delta_GLMB update

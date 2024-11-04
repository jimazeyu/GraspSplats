/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Andreas ten Pas
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef IMAGE_15_CHANNELS_STRATEGY_H_
#define IMAGE_15_CHANNELS_STRATEGY_H_

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <gpd/descriptor/image_geometry.h>
#include <gpd/descriptor/image_strategy.h>

namespace gpd {
namespace descriptor {

/**
 *
 * \brief Calculate 15-channels grasp image.
 *
 * The 15 channels contain height maps and surface normals of the points
 * contained inside the robot hand's closing region from three different views
 * like in the 12-channels image. The three additional channels contain the
 * "shadow" of the points.
 *
 */
class Image15ChannelsStrategy : public ImageStrategy {
 public:
  /**
   * \brief Create a strategy for calculating grasp images.
   * \param image_params the grasp image parameters
   * \param num_threads the number of CPU threads to be used
   * \param num_orientations the number of robot hand orientations
   * \param is_plotting if the images are visualized
   * \return the strategy for calculating grasp images
   */
  Image15ChannelsStrategy(const ImageGeometry &image_params, int num_threads,
                          int num_orientations, bool is_plotting)
      : ImageStrategy(image_params, num_threads, num_orientations,
                      is_plotting) {
    // Calculate shadow length (= max length of shadow required to fill image
    // window).
    Eigen::Vector3d image_dims;
    image_dims << image_params_.depth_, image_params_.height_ / 2.0,
        image_params_.outer_diameter_;
    shadow_length_ = image_dims.maxCoeff();
  }

  /**
   * \brief Create grasp images given a list of grasp candidates.
   * \param hand_set the grasp candidates
   * \param nn_points the point neighborhoods used to calculate the images
   * \return the grasp images
   */
  std::vector<std::unique_ptr<cv::Mat>> createImages(
      const candidate::HandSet &hand_set,
      const util::PointList &nn_points) const;

 protected:
  void createImage(const util::PointList &point_list,
                   const candidate::Hand &hand, const Eigen::Matrix3Xd &shadow,
                   cv::Mat &image) const;

 private:
  cv::Mat calculateImage(const Eigen::Matrix3Xd &points,
                         const Eigen::Matrix3Xd &normals,
                         const Eigen::Matrix3Xd &shadow) const;

  std::vector<cv::Mat> calculateChannels(const Eigen::Matrix3Xd &points,
                                         const Eigen::Matrix3Xd &normals,
                                         const Eigen::Matrix3Xd &shadow) const;

  void showImage(const cv::Mat &image) const;

  double shadow_length_;
};

}  // namespace descriptor
}  // namespace gpd

#endif /* IMAGE_15_CHANNELS_STRATEGY_H_ */

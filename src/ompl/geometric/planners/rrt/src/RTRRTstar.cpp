/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2011, Rice University
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
*   * Neither the name of the Rice University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
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
*********************************************************************/

/* Authors: Alejandro Perez, Sertac Karaman, Ryan Luna, Luis G. Torres, Ioan Sucan, Javier V Gomez, Jonathan Gammell */

#include "ompl/geometric/planners/rrt/RTRRTstar.h"
#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <limits>
#include <vector>
#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>
#include <sys/stat.h>
#include "ompl/base/Goal.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/base/goals/GoalState.h"
#include "ompl/base/goals/GoalStates.h"
#include "ompl/base/goals/GoalLazySamples.h"
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"
#include "ompl/base/samplers/InformedStateSampler.h"
#include "ompl/base/samplers/informed/RejectionInfSampler.h"
#include "ompl/base/samplers/informed/OrderedInfSampler.h"
#include "ompl/tools/config/SelfConfig.h"
#include "ompl/util/GeometricEquations.h"
#include "ompl/base/spaces/RealVectorStateSpace.h"

#include <ros/ros.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <actionlib/client/simple_action_client.h>
#include <std_msgs/Float64MultiArray.h>
#include <ros/package.h>


ompl::geometric::RTRRTstar::RTRRTstar(const base::SpaceInformationPtr &si)
  : base::Planner(si, "RTRRTstar")
{
    specs_.approximateSolutions = true;
    specs_.optimizingPaths = true;
    specs_.canReportIntermediateSolutions = true;

    Planner::declareParam<double>("range", this, &RTRRTstar::setRange, &RTRRTstar::getRange, "0.:1.:10000.");
    Planner::declareParam<double>("goal_bias", this, &RTRRTstar::setGoalBias, &RTRRTstar::getGoalBias, "0.:.05:1.");
    Planner::declareParam<double>("rewire_factor", this, &RTRRTstar::setRewireFactor, &RTRRTstar::getRewireFactor,
                                  "1.0:0.01:2.0");
    Planner::declareParam<bool>("use_k_nearest", this, &RTRRTstar::setKNearest, &RTRRTstar::getKNearest, "0,1");
    Planner::declareParam<bool>("delay_collision_checking", this, &RTRRTstar::setDelayCC, &RTRRTstar::getDelayCC, "0,1");
    Planner::declareParam<bool>("tree_pruning", this, &RTRRTstar::setTreePruning, &RTRRTstar::getTreePruning, "0,1");
    Planner::declareParam<double>("prune_threshold", this, &RTRRTstar::setPruneThreshold, &RTRRTstar::getPruneThreshold,
                                  "0.:.01:1.");
    Planner::declareParam<bool>("pruned_measure", this, &RTRRTstar::setPrunedMeasure, &RTRRTstar::getPrunedMeasure, "0,1");
    Planner::declareParam<bool>("informed_sampling", this, &RTRRTstar::setInformedSampling, &RTRRTstar::getInformedSampling,
                                "0,1");
    Planner::declareParam<bool>("sample_rejection", this, &RTRRTstar::setSampleRejection, &RTRRTstar::getSampleRejection,
                                "0,1");
    Planner::declareParam<bool>("new_state_rejection", this, &RTRRTstar::setNewStateRejection,
                                &RTRRTstar::getNewStateRejection, "0,1");
    Planner::declareParam<bool>("use_admissible_heuristic", this, &RTRRTstar::setAdmissibleCostToCome,
                                &RTRRTstar::getAdmissibleCostToCome, "0,1");
    Planner::declareParam<bool>("ordered_sampling", this, &RTRRTstar::setOrderedSampling, &RTRRTstar::getOrderedSampling,
                                "0,1");
    Planner::declareParam<unsigned int>("ordering_batch_size", this, &RTRRTstar::setBatchSize, &RTRRTstar::getBatchSize,
                                        "1:100:1000000");
    Planner::declareParam<bool>("focus_search", this, &RTRRTstar::setFocusSearch, &RTRRTstar::getFocusSearch, "0,1");
    Planner::declareParam<unsigned int>("number_sampling_attempts", this, &RTRRTstar::setNumSamplingAttempts,
                                        &RTRRTstar::getNumSamplingAttempts, "10:10:100000");

    addPlannerProgressProperty("iterations INTEGER", [this] { return numIterationsProperty(); });
    addPlannerProgressProperty("best cost REAL", [this] { return bestCostProperty(); });

    // Real-time/ROS setup
    /* useKNearest_ = false; */
    new_goal_vec_ = nullptr;
    goal_num_ = 0;
    state_num_ = 0;
    // flag for when motions needs to be checked
    // TODO: need to change the flag when an obstacle is moved/the scene is updated
    scene_changed_ = false;
    /* OMPL_INFORM("Turning tree pruning on"); */
    /* setTreePruning(true); */
    new_goal_sub_ = nh_.subscribe("/new_planner_goal", 100, &ompl::geometric::RTRRTstar::newGoalCallback, this);
    /* scene_changed_sub_ = nh_.subscribe("/move_group/monitored_planning_scene", 100, &ompl::geometric::RTRRTstar::sceneChangedCallback, this); */
    trajectory_client_ = new TrajectoryClient(
        "/execute_trajectory", true);
        /* "/position_joint_trajectory_controller/follow_joint_trajectory", true); */
    while (!trajectory_client_->waitForServer(ros::Duration(1.0))) {
      /* OMPL_INFORM("Waiting for the position_joint_trajectory_controller action server"); */
      OMPL_INFORM("Waiting for the execute_trajectory action server");
    }
    // fill in joint trajectory goal msg
    /* joint_trajectory_goal_.trajectory.joint_names.push_back("panda_joint1"); */
    /* joint_trajectory_goal_.trajectory.joint_names.push_back("panda_joint2"); */
    /* joint_trajectory_goal_.trajectory.joint_names.push_back("panda_joint3"); */
    /* joint_trajectory_goal_.trajectory.joint_names.push_back("panda_joint4"); */
    /* joint_trajectory_goal_.trajectory.joint_names.push_back("panda_joint5"); */
    /* joint_trajectory_goal_.trajectory.joint_names.push_back("panda_joint6"); */
    /* joint_trajectory_goal_.trajectory.joint_names.push_back("panda_joint7"); */
    /* joint_trajectory_goal_.trajectory.joint_trajectory.joint_names.push_back("panda_joint1"); */
    /* joint_trajectory_goal_.trajectory.joint_trajectory.joint_names.push_back("panda_joint2"); */
    /* joint_trajectory_goal_.trajectory.joint_trajectory.joint_names.push_back("panda_joint3"); */
    /* joint_trajectory_goal_.trajectory.joint_trajectory.joint_names.push_back("panda_joint4"); */
    /* joint_trajectory_goal_.trajectory.joint_trajectory.joint_names.push_back("panda_joint5"); */
    /* joint_trajectory_goal_.trajectory.joint_trajectory.joint_names.push_back("panda_joint6"); */
    /* joint_trajectory_goal_.trajectory.joint_trajectory.joint_names.push_back("panda_joint7"); */
}

ompl::geometric::RTRRTstar::~RTRRTstar()
{
  if (xstate_)
    si_->freeState(xstate_);
  if (rmotion_)
  {
    if (rmotion_->state)
        si_->freeState(rmotion_->state);
    delete rmotion_;
  }

  if (trajectory_client_)
    delete trajectory_client_;
  freeMemory();
}

void ompl::geometric::RTRRTstar::setup()
{
  Planner::setup();
  tools::SelfConfig sc(si_, getName());
  sc.configurePlannerRange(maxDistance_); // Originally 6.712660
  OMPL_WARN("maxDistance_: '%f'", maxDistance_);
  if (!si_->getStateSpace()->hasSymmetricDistance() || !si_->getStateSpace()->hasSymmetricInterpolate())
  {
    OMPL_WARN("%s requires a state space with symmetric distance and symmetric interpolation.", getName().c_str());
  }

  if (!nn_)
    nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
  nn_->setDistanceFunction([this](const Motion *a, const Motion *b) { return distanceFunction(a, b); });

  // Setup optimization objective
  //
  // If no optimization objective was specified, then default to
  // optimizing path length as computed by the distance() function
  // in the state space.
  if (pdef_)
  {
    if (pdef_->hasOptimizationObjective())
      opt_ = pdef_->getOptimizationObjective();
    else
    {
      OMPL_INFORM("%s: No optimization objective specified. Defaulting to optimizing path length for the allowed "
                  "planning time.",
                  getName().c_str());
      opt_ = std::make_shared<base::PathLengthOptimizationObjective>(si_);

      // Store the new objective in the problem def'n
      pdef_->setOptimizationObjective(opt_);
    }

    // Set the bestCost_ and prunedCost_ as infinite
    bestCost_ = opt_->infiniteCost();
    prunedCost_ = opt_->infiniteCost();
    // Setup required for moving expandTree outside of solve loop
    // our functor for sorting nearest neighbors
    intermediateSolutionCallback_ptr_ = std::make_unique<const base::ReportIntermediateSolutionFn>(
        pdef_->getIntermediateSolutionCallback());
    compareFn_ptr_ = std::make_unique<CostIndexCompare>(costs_, *opt_);
    symCost_ = opt_->isSymmetric();
    rmotion_ = new Motion(si_);
    rstate_ = rmotion_->state;
    xstate_ = si_->allocState();
  }
  else
  {
    OMPL_INFORM("%s: problem definition is not set, deferring setup completion...", getName().c_str());
    setup_ = false;
  }

  // Get the measure of the entire space:
  prunedMeasure_ = si_->getSpaceMeasure();

  // Calculate some constants:
  calculateRewiringLowerBounds();
}

void ompl::geometric::RTRRTstar::clear()
{
  setup_ = false;
  Planner::clear();
  sampler_.reset();
  infSampler_.reset();
  freeMemory();
  if (nn_)
    nn_->clear();

  bestGoalMotion_ = nullptr;
  goalMotions_.clear();
  startMotions_.clear();

  iterations_ = 0;
  bestCost_ = base::Cost(std::numeric_limits<double>::quiet_NaN());
  prunedCost_ = base::Cost(std::numeric_limits<double>::quiet_NaN());
  prunedMeasure_ = 0.0;
}

void ompl::geometric::RTRRTstar::newGoalCallback(const std_msgs::Float64MultiArray new_goal_msg)
{
  new_goal_vec_ = new std::vector<double>(new_goal_msg.data);
}

void ompl::geometric::RTRRTstar::sceneChangedCallback(const moveit_msgs::PlanningScene planning_scene_msg)
{
  scene_changed_ = planning_scene_msg.is_diff;
}

ompl::base::PlannerStatus ompl::geometric::RTRRTstar::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    goal_ = pdef_->getGoal().get();
    OMPL_INFORM("Switching goal object to GoalState object");
    base::State *gstate_initial = si_->allocState();
    if (base::GoalLazySamples* tmp_gls_initial = dynamic_cast<base::GoalLazySamples*>(goal_))
    {
      while (!tmp_gls_initial->hasStates())
      {
        OMPL_INFORM("Waiting for goal states to be sampled");
        continue;
      }
      // Extract State from GoalStates object
      tmp_gls_initial->sampleGoal(gstate_initial);
      // stop GoalLazySamples sampling thread
      tmp_gls_initial->stopSampling();
    }
    // use setGoalState method to set simplified goal state
    pdef_->setGoalState(gstate_initial);
    goal_ = pdef_->getGoal().get();
    goal_s_ = dynamic_cast<base::GoalSampleableRegion *>(goal_);

    // Check if there are more starts
    if (pis_.haveMoreStartStates() == true)
    {
        // There are, add them
        while (const base::State *st = pis_.nextStart())
        {
            auto *motion = new Motion(si_);
            si_->copyState(motion->state, st);
            motion->cost = opt_->identityCost();
            nn_->add(motion);
            startMotions_.push_back(motion);
        }

        // And assure that, if we're using an informed sampler, it's reset
        infSampler_.reset();
    }
    // No else

    if (nn_->size() == 0)
    {
        OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
        return base::PlannerStatus::INVALID_START;
    }

    // Allocate a sampler if necessary
    if (!sampler_ && !infSampler_)
    {
        allocSampler();
    }

    OMPL_INFORM("%s: Started planning with %u states. Seeking a solution better than %.5f.", getName().c_str(), nn_->size(), opt_->getCostThreshold().value());

    if ((useTreePruning_ || useRejectionSampling_ || useInformedSampling_ || useNewStateRejection_) &&
        !si_->getStateSpace()->isMetricSpace())
        OMPL_WARN("%s: The state space (%s) is not metric and as a result the optimization objective may not satisfy "
                  "the triangle inequality. "
                  "You may need to disable pruning or rejection.",
                  getName().c_str(), si_->getStateSpace()->getName().c_str());


    if (bestGoalMotion_)
        OMPL_INFORM("%s: Starting planning with existing solution of cost %.5f", getName().c_str(),
                    bestCost_.value());

    if (useKNearest_)
        OMPL_INFORM("%s: Initial k-nearest value of %u, k_rrt value: %d", getName().c_str(),
                    (unsigned int)std::ceil(k_rrt_ * log((double)(nn_->size() + 1u))), k_rrt_);
    else
        OMPL_INFORM(
            "%s: Initial rewiring radius of %.2f", getName().c_str(),
            std::min(maxDistance_, r_rrt_ * std::pow(log((double)(nn_->size() + 1u)) / ((double)(nn_->size() + 1u)),
                                                     1 / (double)(si_->getStateDimension()))));

    current_root_ = startMotions_[0];
    OMPL_WARN("Max distance: '%f'", maxDistance_);
    // Realtime variables (should initialize on the stack where possible)
    bool goal_acheived = false;
    Motion *next_motion = nullptr;
    moveit_msgs::ExecuteTrajectoryGoal trajectory_goal_msg;
    rr_time_allowed_ = ros::Duration(0.5);
    ros::Time start_solution_loop = ros::Time::now();
    int motion_addition_count = 0;

    /* while (ptc == false) */
    while (goal_acheived == false)
    {
        iterations_++;

        // Check if our environment (obstacles) or goal state has changed
        ros::spinOnce();
        // If we have a new goal, then change our planner goal state
        if (new_goal_vec_)
        {
          OMPL_WARN("New goal state detected changing Planner goal and removing old solutions");
          base::State *gstate = si_->allocState();
          int i = 0;
          for (std::vector<double>::iterator it = (*new_goal_vec_).begin(); it != (*new_goal_vec_).end(); it++, i++)
          {
            gstate->as<base::RealVectorStateSpace::StateType>()->values[i] = *it;
          }
          pdef_->setGoalState(gstate);
          checkValidity();
          goal_ = pdef_->getGoal().get();
          goal_s_ = dynamic_cast<base::GoalSampleableRegion *>(goal_);
          // Remove previous solutions from goalMotions_ list
          for (std::vector<Motion*>::iterator it = goalMotions_.begin(); it != goalMotions_.end();)
          {
            if (!goal_->isSatisfied((*it)->state))
            {
              OMPL_INFORM("Removing motion from goalMotions_, goalMotions_ size: '%d'", goalMotions_.size());
              (*it)->inGoal = false;
              goalMotions_.erase(it);
            }
            else
              it++;
          }
          OMPL_INFORM("After removing from goalMotions_, goalMotions_ size: '%d'", goalMotions_.size());
          bestGoalMotion_ = nullptr;
          delete new_goal_vec_;
          new_goal_vec_ = nullptr;
        }

        // If we have a path to goal, and the scene changed, then check
        // that the path(s) to goal are collision free
        if (!goalMotions_.empty() && scene_changed_)
        {
          for (std::vector<Motion*>::iterator it = goalMotions_.begin(); it != goalMotions_.end();)
          {
            Motion* check_motion = *it;
            if (!si_->checkMotion(check_motion->state, check_motion->parent->state))
            {
              OMPL_WARN("Motion in goal Motions found to be invalid, removing");
              check_motion->cost = base::Cost(std::numeric_limits<double>::max());
              check_motion->valid = false;
              // Want to find a way to invalidate these nodes without removing them from the tree
              nn_->remove(check_motion);
              goalMotions_.erase(it);
            }
            else
              it++;
            check_motion = check_motion->parent;
            while (check_motion->parent)
            {
              if (!si_->checkMotion(check_motion->state, check_motion->parent->state))
              {
                check_motion->cost = base::Cost(std::numeric_limits<double>::max());
                check_motion->valid = false;
                nn_->remove(check_motion);
              }
              check_motion = check_motion->parent;
            }
            /* if (it == goalMotions_.end()) */
            /*   break; */
          }
          // If the best motion is not valid anymore then change it to a nullptr
          if (bestGoalMotion_ && !bestGoalMotion_->valid)
            bestGoalMotion_ = nullptr;
        }

        // If our trajectory controller is done executing and we have a
        // path to goal, then we can start executing a new trajectory now
        if (trajectory_client_->getState().isDone() && bestGoalMotion_ != nullptr)
        {
          if (next_motion == bestGoalMotion_)
          {
            OMPL_INFORM("GOAL ACHEIVED");
            goal_acheived = true;
            continue;
          }
          next_motion = getNextMotion(bestGoalMotion_);
          sendTrajectoryGoalFromMotion(next_motion);
          storeStepInfo(bestGoalMotion_, next_motion);
          changeRoot(next_motion);
          expandTree(ros::Duration(0.5));
          rewireRoot(ros::Duration(0.5));
        }

        expandTree();


        // terminate if a sufficient solution is found
        if (bestGoalMotion_ && opt_->isSatisfied(bestCost_))
          break;
    }
    // Save goal/path data
    std::string ee_control_path = ros::package::getPath("end_effector_control");
    std::string dir_path = ee_control_path + "/data/planning/";
    std::string base_file_name = "RTRRTstar_run";
    std::string file_ext = ".json";
    std::string file_name = dir_path + base_file_name + file_ext;

    std::ofstream output_file;
    output_file.open(file_name);
    Json::StyledWriter styledWriter;
    output_file << styledWriter.write(path_info_json_);
    output_file.close();

    // Add our solution (if it exists)
    Motion *newSolution = nullptr;
    if (bestGoalMotion_)
    {
        // We have an exact solution
        newSolution = bestGoalMotion_;
    }
    else if (approxGoalMotion_)
    {
        // We don't have a solution, but we do have an approximate solution
        newSolution = approxGoalMotion_;
    }
    // No else, we have nothing

    // Add what we found
    if (newSolution)
    {
        ptc.terminate();
        // construct the solution path
        std::vector<Motion *> mpath;
        Motion *iterMotion = newSolution;
        while (iterMotion != nullptr)
        {
            mpath.push_back(iterMotion);
            iterMotion = iterMotion->parent;
        }
        // set the solution path
        auto path(std::make_shared<PathGeometric>(si_));
        for (int i = mpath.size() - 1; i >= 0; --i)
            path->append(mpath[i]->state);

        // Add the solution path.
        base::PlannerSolution psol(path);
        psol.setPlannerName(getName());

        // If we don't have a goal motion, the solution is approximate
        if (!bestGoalMotion_)
            psol.setApproximate(approxDist_);

        // Does the solution satisfy the optimization objective?
        psol.setOptimized(opt_, newSolution->cost, opt_->isSatisfied(bestCost_));
        pdef_->addSolutionPath(psol);
    }
    // No else, we have nothing

    OMPL_INFORM("%s: Created %u new states. Checked %u rewire options. %u goal states in tree. Final solution cost "
                "%.3f",
                getName().c_str(), statesGenerated_, rewireTest_, goalMotions_.size(), bestCost_.value());

    // We've added a solution if newSolution == true, and it is an approximate solution if bestGoalMotion_ == false
    return {newSolution != nullptr, bestGoalMotion_ == nullptr};
}

void ompl::geometric::RTRRTstar::expandTree(ros::Duration time_to_expand)
{
  expand_tree_start_time_ = ros::Time::now();
  // want a do while loop since we want to expand the tree once if
  // time_to_expand = 0.0 seconds (which it does by default
  do {
    // sample random state (with goal biasing)
    // Goal samples are only sampled until maxSampleCount() goals are in the tree, to prohibit duplicate goal
    // states.
    if (goal_s_ && goalMotions_.size() < goal_s_->maxSampleCount() && rng_.uniform01() < goalBias_ &&
        goal_s_->canSample())
    {
      goal_s_->sampleGoal(rstate_);
    }
    else
    {
      // Attempt to generate a sample, if we fail (e.g., too many rejection attempts), skip the remainder of this
      // loop and return to try again
      if (!sampleUniform(rstate_))
        continue;
    }

    // find closest state in the tree
    Motion *nmotion = nn_->nearest(rmotion_);

    if (*intermediateSolutionCallback_ptr_ && si_->equalStates(nmotion->state, rstate_))
      continue;

    base::State *dstate = rstate_;

    // find state to add to the tree
    double d = si_->distance(nmotion->state, rstate_);
    if (d > maxDistance_)
    {
      si_->getStateSpace()->interpolate(nmotion->state, rstate_, maxDistance_ / d, xstate_);
      dstate = xstate_;
    }

    // Check if the motion between the nearest state and the state to add is valid
    if (si_->checkMotion(nmotion->state, dstate))
    {
      // create a motion
      auto *motion = new Motion(si_);
      si_->copyState(motion->state, dstate);
      motion->parent = nmotion;
      motion->incCost = opt_->motionCost(nmotion->state, motion->state);
      motion->cost = opt_->combineCosts(nmotion->cost, motion->incCost);

      // Find nearby neighbors of the new motion
      getNeighbors(motion, nbh_);

      rewireTest_ += nbh_.size();
      ++statesGenerated_;

      // cache for distance computations
      //
      // Our cost caches only increase in size, so they're only
      // resized if they can't fit the current neighborhood
      if (costs_.size() < nbh_.size())
      {
        costs_.resize(nbh_.size());
        incCosts_.resize(nbh_.size());
        sortedCostIndices_.resize(nbh_.size());
      }

      // cache for motion validity (only useful in a symmetric space)
      //
      // Our validity caches only increase in size, so they're
      // only resized if they can't fit the current neighborhood
      if (valid_.size() < nbh_.size())
        valid_.resize(nbh_.size());
      std::fill(valid_.begin(), valid_.begin() + nbh_.size(), 0);

      // Finding the nearest neighbor to connect to
      // By default, neighborhood states are sorted by cost, and collision checking
      // is performed in increasing order of cost
      if (delayCC_)
      {
        // calculate all costs_ and distances
        for (std::size_t i = 0; i < nbh_.size(); ++i)
        {
          incCosts_[i] = opt_->motionCost(nbh_[i]->state, motion->state);
          costs_[i] = opt_->combineCosts(nbh_[i]->cost, incCosts_[i]);
        }

        // sort the nodes
        //
        // we're using index-value pairs so that we can get at
        // original, unsorted indices
        for (std::size_t i = 0; i < nbh_.size(); ++i)
          sortedCostIndices_[i] = i;
        /* std::sort(sortedCostIndices_.begin(), sortedCostIndices_.begin() + nbh_.size(), compareFn); */
        std::sort(sortedCostIndices_.begin(), sortedCostIndices_.begin() + nbh_.size(), *compareFn_ptr_);

        // collision check until a valid motion is found
        //
        // ASYMMETRIC CASE: it's possible that none of these
        // neighbors are valid. This is fine, because motion
        // already has a connection to the tree through
        // nmotion (with populated cost fields!).
        for (std::vector<std::size_t>::const_iterator i = sortedCostIndices_.begin();
             i != sortedCostIndices_.begin() + nbh_.size(); ++i)
        {
          if (nbh_[*i] == nmotion || ((!useKNearest_ || si_->distance(nbh_[*i]->state, motion->state) < maxDistance_) &&
                si_->checkMotion(nbh_[*i]->state, motion->state)))
          {
            motion->incCost = incCosts_[*i];
            motion->cost = costs_[*i];
            motion->parent = nbh_[*i];
            valid_[*i] = 1;
            break;
          }
          else
            valid_[*i] = -1;
        }
      }
      else  // if not delayCC
      {
        motion->incCost = opt_->motionCost(nmotion->state, motion->state);
        motion->cost = opt_->combineCosts(nmotion->cost, motion->incCost);
        // find which one we connect the new state to
        for (std::size_t i = 0; i < nbh_.size(); ++i)
        {
          if (nbh_[i] != nmotion)
          {
            incCosts_[i] = opt_->motionCost(nbh_[i]->state, motion->state);
            costs_[i] = opt_->combineCosts(nbh_[i]->cost, incCosts_[i]);
            if (opt_->isCostBetterThan(costs_[i], motion->cost))
            {
              if ((!useKNearest_ || si_->distance(nbh_[i]->state, motion->state) < maxDistance_) &&
                  si_->checkMotion(nbh_[i]->state, motion->state))
              {
                motion->incCost = incCosts_[i];
                motion->cost = costs_[i];
                motion->parent = nbh_[i];
                valid_[i] = 1;
              }
              else
                valid_[i] = -1;
            }
          }
          else
          {
            incCosts_[i] = motion->incCost;
            costs_[i] = motion->cost;
            valid_[i] = 1;
          }
        }
      }

      if (useNewStateRejection_)
      {
        if (opt_->isCostBetterThan(solutionHeuristic(motion), bestCost_))
        {
          nn_->add(motion);
          motion->parent->children.push_back(motion);
        }
        else  // If the new motion does not improve the best cost it is ignored.
        {
          si_->freeState(motion->state);
          delete motion;
          continue;
        }
      }
      else
      {
        // add motion to the tree
        nn_->add(motion);
        motion->parent->children.push_back(motion);
      }

      bool checkForSolution = false;
      for (std::size_t i = 0; i < nbh_.size(); ++i)
      {
        if (nbh_[i] != motion->parent)
        {
          base::Cost nbhIncCost;
          if (symCost_)
            nbhIncCost = incCosts_[i];
          else
            nbhIncCost = opt_->motionCost(motion->state, nbh_[i]->state);
          base::Cost nbhNewCost = opt_->combineCosts(motion->cost, nbhIncCost);
          if (opt_->isCostBetterThan(nbhNewCost, nbh_[i]->cost))
          {
            bool motionValid;
            if (valid_[i] == 0)
            {
              motionValid =
                (!useKNearest_ || si_->distance(nbh_[i]->state, motion->state) < maxDistance_) &&
                si_->checkMotion(motion->state, nbh_[i]->state);
            }
            else
            {
              motionValid = (valid_[i] == 1);
            }

            if (motionValid)
            {
              // Remove this node from its parent list
              removeFromParent(nbh_[i]);

              // Add this node to the new parent
              nbh_[i]->parent = motion;
              nbh_[i]->incCost = nbhIncCost;
              nbh_[i]->cost = nbhNewCost;
              nbh_[i]->parent->children.push_back(nbh_[i]);

              // Update the costs_ of the node's children
              updateChildCosts(nbh_[i]);

              checkForSolution = true;
            }
          }
        }
      }

      // Add the new motion to the goalMotion_ list, if it satisfies the goal
      double distanceFromGoal;
      if (goal_->isSatisfied(motion->state, &distanceFromGoal))
      {
        motion->inGoal = true;
        goalMotions_.push_back(motion);
        OMPL_INFORM("DWY: adding to goalMotions_. New goalMotions_ size: '%d'", goalMotions_.size());
        checkForSolution = true;
      }

      // Checking for solution or iterative improvement
      if (checkForSolution)
      {
        bool updatedSolution = false;
        if (!bestGoalMotion_ && !goalMotions_.empty())
        {
          // We have found our first solution, store it as the best. We only add one
          // vertex at a time, so there can only be one goal vertex at this moment.
          bestGoalMotion_ = goalMotions_.front();
          bestCost_ = bestGoalMotion_->cost;
          updatedSolution = true;

          OMPL_INFORM("%s: Found an initial solution with a cost of %.2f in %u iterations (%u "
                      "vertices in the graph)",
                      getName().c_str(), bestCost_.value(), iterations_, nn_->size());
        }
        else
        {
          // We already have a solution, iterate through the list of goal vertices
          // and see if there's any improvement.
          for (auto &goalMotion : goalMotions_)
          {
            // Is this goal motion better than the (current) best?
            if (opt_->isCostBetterThan(goalMotion->cost, bestCost_))
            {
              bestGoalMotion_ = goalMotion;
              bestCost_ = bestGoalMotion_->cost;
              updatedSolution = true;
              OMPL_INFORM("Found improved solution with cost '%.2f'", bestCost_.value());

              // Check if it satisfies the optimization objective, if it does, break the for loop
              if (opt_->isSatisfied(bestCost_))
              {
                break;
              }
            }
          }
        }

        if (updatedSolution)
        {
          /* OMPL_INFORM("Updated solution: outputting bestGoalMotion_->state"); */
          /* for (int i=0; i<8; i++) */
          /* { */
          /*   OMPL_INFORM("rstate[%d]: '%f'", i, bestGoalMotion_->state->as<base::RealVectorStateSpace::StateType>()->values[i]); */
          /* } */
          if (useTreePruning_)
          {
            pruneTree(bestCost_);
          }

          if (*intermediateSolutionCallback_ptr_)
          {
            std::vector<const base::State *> spath;
            Motion *intermediate_solution =
                bestGoalMotion_->parent;  // Do not include goal state to simplify code.

            // Push back until we find the start, but not the start itself
            while (intermediate_solution->parent != nullptr)
            {
              spath.push_back(intermediate_solution->state);
              intermediate_solution = intermediate_solution->parent;
            }

            (*intermediateSolutionCallback_ptr_)(this, spath, bestCost_);
          }
        }
      }

      // Checking for approximate solution (closest state found to the goal)
      if (goalMotions_.size() == 0 && distanceFromGoal < approxDist_)
      {
        approxGoalMotion_ = motion;
        approxDist_ = distanceFromGoal;
      }
    }
  }
  while (ros::Time::now() - expand_tree_start_time_ > time_to_expand);
  // TODO: add some logging for number of expansion iterations that occured...
}

void ompl::geometric::RTRRTstar::changeRoot(Motion *new_root)
{
  // make the previous root a child of this new root
  Motion *prev_root = new_root->parent;
  new_root->children.push_back(prev_root);
  prev_root->parent = new_root;
  // remove the new root from the children of the previous root
  removeFromParent(new_root);
  new_root->parent = nullptr;
  // Re-calculate costs_
  new_root->cost = opt_->identityCost();
  prev_root->incCost = opt_->motionCost(new_root->state, prev_root->state);
  prev_root->cost = opt_->combineCosts(new_root->cost, prev_root->incCost);
  current_root_ = new_root;
  updateChildCosts(new_root);
  // add the new current motion to the root rewiring queue so the neighboring nodes move along
  // with the root.
  /* rootRewireQueue_.push_front(new_root); */
}

void ompl::geometric::RTRRTstar::rewireRoot(ros::Duration time_to_rewire)
{
  std::deque<Motion *> rootRewireQueue;
  std::set<Motion *> rootRewireSet;

  rootRewireQueue.push_front(current_root_);
  rootRewireSet.insert(current_root_);

  root_rewire_start_time_ = ros::Time::now();
  int iterations = 0;
  int toggle = 0;

  while (!rootRewireQueue.empty())
  {
    // if time_to_rewire is 0.0 seconds (default), then rewire until
    // queue is empty
    // TODO: test what happens to cost outcome if I allow rewiring to complete fully
    if (time_to_rewire != ros::Duration(0.0) && ros::Time::now() - root_rewire_start_time_ > time_to_rewire)
      break;
    iterations++;
    rr_motion_ = rootRewireQueue.front();
    rootRewireQueue.pop_front();
    getNeighbors(rr_motion_, rr_nbh_);

    int unique_count = 0;
    for (std::size_t k=0; k<rr_nbh_.size(); k++)
    {
      if (!rootRewireSet.count(rr_nbh_[k]))
        unique_count++;
    }
    if (unique_count > 0)
    {
      OMPL_INFORM("iteration '%d' gathered '%d' UNIQUE motions", iterations, unique_count);
      OMPL_INFORM("Size of tree: '%d'", rr_nbh_.size());
    }

    for (std::size_t i = 0; i < rr_nbh_.size(); ++i)
    {
      if (rr_nbh_[i] != rr_motion_->parent)
      {
        base::Cost rrNbhIncCost = opt_->motionCost(rr_motion_->state, rr_nbh_[i]->state);
        base::Cost rrNbhNewCost = opt_->combineCosts(rr_motion_->cost, rrNbhIncCost);
        if (opt_->isCostBetterThan(rrNbhNewCost, rr_nbh_[i]->cost))
        {
          if ((!useKNearest_ || si_->distance(rr_nbh_[i]->state, rr_motion_->state) < maxDistance_) &&
              si_->checkMotion(rr_motion_->state, rr_nbh_[i]->state))
          {
            // Remove this node from its parent list
            removeFromParent(rr_nbh_[i]);

            // Add this node to the new parent
            rr_nbh_[i]->parent = rr_motion_;
            rr_nbh_[i]->incCost = rrNbhIncCost;
            rr_nbh_[i]->cost = rrNbhNewCost;
            rr_nbh_[i]->parent->children.push_back(rr_nbh_[i]);

            // Update the costs_ of the node's children
            updateChildCosts(rr_nbh_[i]);

            // TODO: should I set checkForSolution = true here?
          }
        }
      }
      if (rootRewireSet.insert(rr_nbh_[i]).second == true)
      {
        if (toggle != 1)
        {
          OMPL_INFORM("SWITCH: INSERTING INTO ROOT REWIRE SET. iteration: '%d'", iterations);
          toggle = 1;
        }
        rootRewireQueue.push_back(rr_nbh_[i]);
      }
      else // rootRewireSet.insert(rr_nbh_[i]).second = false
      {
        if (toggle != 2)
        {
          OMPL_INFORM("SWITCH: NOT INSERTING INTO ROOT REWIRE SET. iteration: '%d'", iterations);
          toggle = 2;
        }
      }
    }
  }
  OMPL_INFORM("Root rewire completed '%d' iterations", iterations);
  OMPL_INFORM("Root rewire queue size: '%d'", rootRewireQueue.size());
}

inline bool ompl::geometric::RTRRTstar::fileExists(const std::string& name)
{
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

void ompl::geometric::RTRRTstar::storeStepInfo(Motion *goal, Motion *next)
{
  base::State *goal_state = goal->state;
  base::State *next_state = next->state;
  if (goal != prev_goal_)
  {
    goal_num_++;
    state_num_ = 0;
    Json::Value goal_arr(Json::arrayValue);
    for (int i=0; i<7; i++)
    {
      goal_arr.append(goal_state->as<base::RealVectorStateSpace::StateType>()->values[i]);
    }
    path_info_json_["Goals"][goal_num_-1] = goal_arr;
    prev_goal_ = goal;
  }

  // Have to add the previous state as the start state if we're starting a new path
  if (state_num_ == 0)
  {
    base::State *prev_state = next->parent->state;
    Json::Value step_from_arr(Json::arrayValue);
    for (int j=0; j<7; j++)
    {
      step_from_arr.append(prev_state->as<base::RealVectorStateSpace::StateType>()->values[j]);
    }
    path_info_json_["Goal" + std::to_string(goal_num_) + "States"][state_num_] = step_from_arr;
    state_num_++;
  }

  Json::Value step_to_arr(Json::arrayValue);
  for (int k=0; k<7; k++)
  {
    step_to_arr.append(next_state->as<base::RealVectorStateSpace::StateType>()->values[k]);
  }
  path_info_json_["Goal" + std::to_string(goal_num_) + "States"][state_num_] = step_to_arr;
  state_num_++;
}

ompl::geometric::RTRRTstar::Motion* ompl::geometric::RTRRTstar::getNextMotion(Motion *last_motion)
{
  Motion *next_motion = last_motion;
  while (next_motion->parent->parent != nullptr)
  {
    next_motion = next_motion->parent;
  }
  return next_motion;
}

void ompl::geometric::RTRRTstar::sendTrajectoryGoalFromMotion(Motion *next_motion)
{
  moveit_msgs::ExecuteTrajectoryGoal joint_trajectory_goal;
  joint_trajectory_goal.trajectory.joint_trajectory.joint_names.push_back("panda_joint1");
  joint_trajectory_goal.trajectory.joint_trajectory.joint_names.push_back("panda_joint2");
  joint_trajectory_goal.trajectory.joint_trajectory.joint_names.push_back("panda_joint3");
  joint_trajectory_goal.trajectory.joint_trajectory.joint_names.push_back("panda_joint4");
  joint_trajectory_goal.trajectory.joint_trajectory.joint_names.push_back("panda_joint5");
  joint_trajectory_goal.trajectory.joint_trajectory.joint_names.push_back("panda_joint6");
  joint_trajectory_goal.trajectory.joint_trajectory.joint_names.push_back("panda_joint7");
  /* joint_trajectory_goal_.trajectory.points.resize(1); */
  /* joint_trajectory_goal_.trajectory.points[0].positions.resize(7); */
  /* joint_trajectory_goal_.trajectory.points[0].velocities.resize(7); */
  /* joint_trajectory_goal_.trajectory.joint_trajectory.points.resize(1); */
  joint_trajectory_goal.trajectory.joint_trajectory.points.resize(2);
  joint_trajectory_goal.trajectory.joint_trajectory.points[0].positions.resize(7);
  joint_trajectory_goal.trajectory.joint_trajectory.points[1].positions.resize(7);
  joint_trajectory_goal.trajectory.joint_trajectory.points[0].velocities.resize(7);
  joint_trajectory_goal.trajectory.joint_trajectory.points[1].velocities.resize(7);

  base::State *start_state = next_motion->parent->state;
  base::State *next_state = next_motion->state;
  for (int j=0; j<7; j++)
  {
    /* joint_trajectory_goal_.trajectory.points[0].positions[j] = */
    joint_trajectory_goal.trajectory.joint_trajectory.points[0].positions[j] =
      start_state->as<base::RealVectorStateSpace::StateType>()->values[j];
    joint_trajectory_goal.trajectory.joint_trajectory.points[1].positions[j] =
      next_state->as<base::RealVectorStateSpace::StateType>()->values[j];
    /* joint_trajectory_goal_.trajectory.points[0].velocities[j] = 0.0; */
    joint_trajectory_goal.trajectory.joint_trajectory.points[0].velocities[j] = 0.0;
    joint_trajectory_goal.trajectory.joint_trajectory.points[1].velocities[j] = 0.0;
  }
  /* joint_trajectory_goal_.trajectory.points[0].time_from_start = ros::Duration(0.0); */
  /* joint_trajectory_goal_.trajectory.header.stamp = ros::Time::now() + ros::Duration(0.50); */
  joint_trajectory_goal.trajectory.joint_trajectory.points[0].time_from_start = ros::Duration(0.0);
  joint_trajectory_goal.trajectory.joint_trajectory.points[1].time_from_start = ros::Duration(1.0);
  joint_trajectory_goal.trajectory.joint_trajectory.header.stamp = ros::Time::now() + ros::Duration(0.1);

  trajectory_client_->sendGoal(joint_trajectory_goal);
}

void ompl::geometric::RTRRTstar::evalRoot(Motion *goal)
{
  Motion *root = goal;
  while (root->parent)
    root = root->parent;
  printStateValues(root->state);
}

void ompl::geometric::RTRRTstar::printStateValues(const ompl::base::State *state)
{
  for (int i=0; i<7; i++)
  {
    OMPL_INFORM("State '%d': '%f'", i, state->as<base::RealVectorStateSpace::StateType>()->values[i]);
  }
}

void ompl::geometric::RTRRTstar::getNeighbors(Motion *motion, std::vector<Motion *> &nbh_) const
{
  auto cardDbl = static_cast<double>(nn_->size() + 1u);
  if (useKNearest_)
  {
    //- k-nearest RRT*
    unsigned int k = std::ceil(k_rrt_ * log(cardDbl));
    /* OMPL_INFORM("k = '%d', k_rrt_ = '%f', cardDbl = '%f', log(cardDbl) = '%f'", k, k_rrt_, cardDbl, log(cardDbl)); */
    nn_->nearestK(motion, k, nbh_);
  }
  else
  {
    double r = std::min(
      maxDistance_, r_rrt_ * std::pow(log(cardDbl) / cardDbl, 1 / static_cast<double>(si_->getStateDimension())));
    /* nn_->nearestR(motion, r, nbh_); */
    nn_->nearestR(motion, maxDistance_, nbh_);
  }
}

void ompl::geometric::RTRRTstar::removeFromParent(Motion *m)
{
  for (auto it = m->parent->children.begin(); it != m->parent->children.end(); ++it)
  {
    if (*it == m)
    {
      m->parent->children.erase(it);
      break;
    }
  }
}

void ompl::geometric::RTRRTstar::updateChildCosts(Motion *m)
{
  for (std::size_t i = 0; i < m->children.size(); ++i)
  {
    if (m->valid)
    {
      m->children[i]->cost = opt_->combineCosts(m->cost, m->children[i]->incCost);
      updateChildCosts(m->children[i]);
    }
    else
    {
      m->children[i]->cost = base::Cost(std::numeric_limits<double>::max());
      m->children[i]->valid = false;
    }
  }
}

void ompl::geometric::RTRRTstar::freeMemory()
{
  OMPL_WARN("DWYDBG: freeMemory called in RTRRTstar");
  if (nn_)
  {
    std::vector<Motion *> motions;
    nn_->list(motions);
    for (auto &motion : motions)
    {
      if (motion->state)
        si_->freeState(motion->state);
      delete motion;
    }
  }
}

void ompl::geometric::RTRRTstar::getPlannerData(base::PlannerData &data) const
{
  Planner::getPlannerData(data);

  std::vector<Motion *> motions;
  if (nn_)
    nn_->list(motions);

  if (bestGoalMotion_)
    data.addGoalVertex(base::PlannerDataVertex(bestGoalMotion_->state));

  for (auto &motion : motions)
  {
    if (motion->parent == nullptr)
      data.addStartVertex(base::PlannerDataVertex(motion->state));
    else
      data.addEdge(base::PlannerDataVertex(motion->parent->state), base::PlannerDataVertex(motion->state));
  }
}

int ompl::geometric::RTRRTstar::pruneTree(const base::Cost &pruneTreeCost)
{
  // Variable
  // The percent improvement (expressed as a [0,1] fraction) in cost
  double fracBetter;
  // The number pruned
  int numPruned = 0;

  if (opt_->isFinite(prunedCost_))
  {
    fracBetter = std::abs((pruneTreeCost.value() - prunedCost_.value()) / prunedCost_.value());
  }
  else
  {
    fracBetter = 1.0;
  }

  if (fracBetter > pruneThreshold_)
  {
    // We are only pruning motions if they, AND all descendents, have a estimated cost greater than pruneTreeCost
    // The easiest way to do this is to find leaves that should be pruned and ascend up their ancestry until a
    // motion is found that is kept.
    // To avoid making an intermediate copy of the NN structure, we process the tree by descending down from the
    // start(s).
    // In the first pass, all Motions with a cost below pruneTreeCost, or Motion's with children with costs_ below
    // pruneTreeCost are added to the replacement NN structure,
    // while all other Motions are stored as either a 'leaf' or 'chain' Motion. After all the leaves are
    // disconnected and deleted, we check
    // if any of the the chain Motions are now leaves, and repeat that process until done.
    // This avoids (1) copying the NN structure into an intermediate variable and (2) the use of the expensive
    // NN::remove() method.

    // Variable
    // The queue of Motions to process:
    std::queue<Motion *, std::deque<Motion *>> motionQueue;
    // The list of leaves to prune
    std::queue<Motion *, std::deque<Motion *>> leavesToPrune;
    // The list of chain vertices to recheck after pruning
    std::list<Motion *> chainsToRecheck;

    // Clear the NN structure:
    nn_->clear();

    // Put all the starts into the NN structure and their children into the queue:
    // We do this so that start states are never pruned.
    for (auto &startMotion : startMotions_)
    {
      // Add to the NN
      nn_->add(startMotion);

      // Add their children to the queue:
      addChildrenToList(&motionQueue, startMotion);
    }

    while (motionQueue.empty() == false)
    {
      // Test, can the current motion ever provide a better solution?
      if (keepCondition(motionQueue.front(), pruneTreeCost))
      {
        // Yes it can, so it definitely won't be pruned
        // Add it back into the NN structure
        nn_->add(motionQueue.front());

        // Add it's children to the queue
        addChildrenToList(&motionQueue, motionQueue.front());
      }
      else
      {
        // No it can't, but does it have children?
        if (motionQueue.front()->children.empty() == false)
        {
          // Yes it does.
          // We can minimize the number of intermediate chain motions if we check their children
          // If any of them won't be pruned, then this motion won't either. This intuitively seems
          // like a nice balance between following the descendents forever.

          // Variable
          // Whether the children are definitely to be kept.
          bool keepAChild = false;

          // Find if any child is definitely not being pruned.
          for (unsigned int i = 0u; keepAChild == false && i < motionQueue.front()->children.size(); ++i)
          {
            // Test if the child can ever provide a better solution
            keepAChild = keepCondition(motionQueue.front()->children.at(i), pruneTreeCost);
          }

          // Are we *definitely* keeping any of the children?
          if (keepAChild)
          {
            // Yes, we are, so we are not pruning this motion
            // Add it back into the NN structure.
            nn_->add(motionQueue.front());
          }
          else
          {
            // No, we aren't. This doesn't mean we won't though
            // Move this Motion to the temporary list
            chainsToRecheck.push_back(motionQueue.front());
          }

          // Either way. add it's children to the queue
          addChildrenToList(&motionQueue, motionQueue.front());
        }
        else
        {
          // No, so we will be pruning this motion:
          leavesToPrune.push(motionQueue.front());
        }
      }

      // Pop the iterator, std::list::erase returns the next iterator
      motionQueue.pop();
    }

    // We now have a list of Motions to definitely remove, and a list of Motions to recheck
    // Iteratively check the two lists until there is nothing to to remove
    while (leavesToPrune.empty() == false)
    {
      // First empty the current leaves-to-prune
      while (leavesToPrune.empty() == false)
      {
        // If this leaf is a goal, remove it from the goal set
        if (leavesToPrune.front()->inGoal == true)
        {
          // Warn if pruning the _best_ goal
          if (leavesToPrune.front() == bestGoalMotion_)
          {
            OMPL_ERROR("%s: Pruning the best goal.", getName().c_str());
          }
          // Remove it
          goalMotions_.erase(std::remove(goalMotions_.begin(), goalMotions_.end(), leavesToPrune.front()),
                               goalMotions_.end());
        }

        // Remove the leaf from its parent
        removeFromParent(leavesToPrune.front());

        // Erase the actual motion
        // First free the state
        si_->freeState(leavesToPrune.front()->state);

        // then delete the pointer
        delete leavesToPrune.front();

        // And finally remove it from the list, erase returns the next iterator
        leavesToPrune.pop();

        // Update our counter
        ++numPruned;
      }

      // Now, we need to go through the list of chain vertices and see if any are now leaves
      auto mIter = chainsToRecheck.begin();
      while (mIter != chainsToRecheck.end())
      {
        // Is the Motion a leaf?
        if ((*mIter)->children.empty() == true)
        {
          // It is, add to the removal queue
          leavesToPrune.push(*mIter);

          // Remove from this queue, getting the next
          mIter = chainsToRecheck.erase(mIter);
        }
        else
        {
          // Is isn't, skip to the next
          ++mIter;
        }
      }
    }

    // Now finally add back any vertices left in chainsToReheck.
    // These are chain vertices that have descendents that we want to keep
    for (const auto &r : chainsToRecheck)
      // Add the motion back to the NN struct:
      nn_->add(r);

    // All done pruning.
    // Update the cost at which we've pruned:
    prunedCost_ = pruneTreeCost;

    // And if we're using the pruned measure, the measure to which we've pruned
    if (usePrunedMeasure_)
    {
      prunedMeasure_ = infSampler_->getInformedMeasure(prunedCost_);

      if (useKNearest_ == false)
      {
        calculateRewiringLowerBounds();
      }
    }
    // No else, prunedMeasure_ is the si_ measure by default.
  }

  return numPruned;
}

void ompl::geometric::RTRRTstar::addChildrenToList(std::queue<Motion *, std::deque<Motion *>> *motionList, Motion *motion)
{
  for (auto &child : motion->children)
  {
    motionList->push(child);
  }
}

bool ompl::geometric::RTRRTstar::keepCondition(const Motion *motion, const base::Cost &threshold) const
{
  // We keep if the cost-to-come-heuristic of motion is <= threshold, by checking
  // if !(threshold < heuristic), as if b is not better than a, then a is better than, or equal to, b
  if (bestGoalMotion_ && motion == bestGoalMotion_)
  {
    // If the threshold is the theoretical minimum, the bestGoalMotion_ will sometimes fail the test due to floating point precision. Avoid that.
    return true;
  }

  return !opt_->isCostBetterThan(threshold, solutionHeuristic(motion));
}

ompl::base::Cost ompl::geometric::RTRRTstar::solutionHeuristic(const Motion *motion) const
{
  base::Cost costToCome;
  if (useAdmissibleCostToCome_)
  {
    // Start with infinite cost
    costToCome = opt_->infiniteCost();

    // Find the min from each start
    for (auto &startMotion : startMotions_)
    {
      // lower-bounding cost from the start to the state
      costToCome = opt_->betterCost(costToCome, opt_->motionCost(startMotion->state, motion->state));
    }
  }
  else
  {
    costToCome = motion->cost;  // current cost from the state to the goal
  }

  const base::Cost costToGo =
    opt_->costToGo(motion->state, pdef_->getGoal().get());  // lower-bounding cost from the state to the goal
  return opt_->combineCosts(costToCome, costToGo);            // add the two costs_
}

void ompl::geometric::RTRRTstar::setTreePruning(const bool prune)
{
  if (static_cast<bool>(opt_) == true)
  {
    if (opt_->hasCostToGoHeuristic() == false)
    {
      OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
    }
  }

  // If we just disabled tree pruning, but we wee using prunedMeasure, we need to disable that as it required myself
  if (prune == false && getPrunedMeasure() == true)
  {
    setPrunedMeasure(false);
  }

  // Store
  useTreePruning_ = prune;
}

void ompl::geometric::RTRRTstar::setPrunedMeasure(bool informedMeasure)
{
  if (static_cast<bool>(opt_) == true)
  {
    if (opt_->hasCostToGoHeuristic() == false)
    {
      OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
    }
  }

  // This option only works with informed sampling
  if (informedMeasure == true && (useInformedSampling_ == false || useTreePruning_ == false))
  {
    OMPL_ERROR("%s: InformedMeasure requires InformedSampling and TreePruning.", getName().c_str());
  }

  // Check if we're changed and update parameters if we have:
  if (informedMeasure != usePrunedMeasure_)
  {
    // Store the setting
    usePrunedMeasure_ = informedMeasure;

    // Update the prunedMeasure_ appropriately, if it has been configured.
    if (setup_ == true)
    {
      if (usePrunedMeasure_)
      {
        prunedMeasure_ = infSampler_->getInformedMeasure(prunedCost_);
      }
      else
      {
        prunedMeasure_ = si_->getSpaceMeasure();
      }
    }

    // And either way, update the rewiring radius if necessary
    if (useKNearest_ == false)
    {
      calculateRewiringLowerBounds();
    }
  }
}

void ompl::geometric::RTRRTstar::setInformedSampling(bool informedSampling)
{
  if (static_cast<bool>(opt_) == true)
  {
    if (opt_->hasCostToGoHeuristic() == false)
    {
      OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
    }
  }

  // This option is mutually exclusive with setSampleRejection, assert that:
  if (informedSampling == true && useRejectionSampling_ == true)
  {
    OMPL_ERROR("%s: InformedSampling and SampleRejection are mutually exclusive options.", getName().c_str());
  }

  // If we just disabled tree pruning, but we are using prunedMeasure, we need to disable that as it required myself
  if (informedSampling == false && getPrunedMeasure() == true)
  {
    setPrunedMeasure(false);
  }

  // Check if we're changing the setting of informed sampling. If we are, we will need to create a new sampler, which
  // we only want to do if one is already allocated.
  if (informedSampling != useInformedSampling_)
  {
    // If we're disabled informedSampling, and prunedMeasure is enabled, we need to disable that
    if (informedSampling == false && usePrunedMeasure_ == true)
    {
      setPrunedMeasure(false);
    }

    // Store the value
    useInformedSampling_ = informedSampling;

    // If we currently have a sampler, we need to make a new one
    if (sampler_ || infSampler_)
    {
      // Reset the samplers
      sampler_.reset();
      infSampler_.reset();

      // Create the sampler
      allocSampler();
    }
  }
}

void ompl::geometric::RTRRTstar::setSampleRejection(const bool reject)
{
  if (static_cast<bool>(opt_) == true)
  {
    if (opt_->hasCostToGoHeuristic() == false)
    {
      OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
    }
  }

  // This option is mutually exclusive with setInformedSampling, assert that:
  if (reject == true && useInformedSampling_ == true)
  {
    OMPL_ERROR("%s: InformedSampling and SampleRejection are mutually exclusive options.", getName().c_str());
  }

  // Check if we're changing the setting of rejection sampling. If we are, we will need to create a new sampler, which
  // we only want to do if one is already allocated.
  if (reject != useRejectionSampling_)
  {
    // Store the setting
    useRejectionSampling_ = reject;

    // If we currently have a sampler, we need to make a new one
    if (sampler_ || infSampler_)
    {
      // Reset the samplers
      sampler_.reset();
      infSampler_.reset();

      // Create the sampler
      allocSampler();
    }
  }
}

void ompl::geometric::RTRRTstar::setOrderedSampling(bool orderSamples)
{
  // Make sure we're using some type of informed sampling
  if (useInformedSampling_ == false && useRejectionSampling_ == false)
  {
    OMPL_ERROR("%s: OrderedSampling requires either informed sampling or rejection sampling.", getName().c_str());
  }

  // Check if we're changing the setting. If we are, we will need to create a new sampler, which we only want to do if
  // one is already allocated.
  if (orderSamples != useOrderedSampling_)
  {
    // Store the setting
    useOrderedSampling_ = orderSamples;

    // If we currently have a sampler, we need to make a new one
    if (sampler_ || infSampler_)
    {
      // Reset the samplers
      sampler_.reset();
      infSampler_.reset();

      // Create the sampler
      allocSampler();
    }
  }
}

void ompl::geometric::RTRRTstar::allocSampler()
{
  // Allocate the appropriate type of sampler.
  if (useInformedSampling_)
  {
    // We are using informed sampling, this can end-up reverting to rejection sampling in some cases
    OMPL_INFORM("%s: Using informed sampling.", getName().c_str());
    infSampler_ = opt_->allocInformedStateSampler(pdef_, numSampleAttempts_);
  }
  else if (useRejectionSampling_)
  {
    // We are explicitly using rejection sampling.
    OMPL_INFORM("%s: Using rejection sampling.", getName().c_str());
    infSampler_ = std::make_shared<base::RejectionInfSampler>(pdef_, numSampleAttempts_);
  }
  else
  {
    // We are using a regular sampler
    sampler_ = si_->allocStateSampler();
  }

  // Wrap into a sorted sampler
  if (useOrderedSampling_ == true)
  {
    infSampler_ = std::make_shared<base::OrderedInfSampler>(infSampler_, batchSize_);
  }
  // No else
}

bool ompl::geometric::RTRRTstar::sampleUniform(base::State *statePtr)
{
  // Use the appropriate sampler
  if (useInformedSampling_ || useRejectionSampling_)
  {
    // Attempt the focused sampler and return the result.
    // If bestCost is changing a lot by small amounts, this could
    // be prunedCost_ to reduce the number of times the informed sampling
    // transforms are recalculated.
    return infSampler_->sampleUniform(statePtr, bestCost_);
  }
  else
  {
    // Simply return a state from the regular sampler
    sampler_->sampleUniform(statePtr);

    // Always true
    return true;
  }
}

void ompl::geometric::RTRRTstar::calculateRewiringLowerBounds()
{
  const auto dimDbl = static_cast<double>(si_->getStateDimension());

  // k_rrt > 2^(d + 1) * e * (1 + 1 / d).  K-nearest RRT*
  k_rrt_ = rewireFactor_ * (std::pow(2, dimDbl + 1) * boost::math::constants::e<double>() * (1.0 + 1.0 / dimDbl));

  // r_rrt > (2*(1+1/d))^(1/d)*(measure/ballvolume)^(1/d)
  // If we're not using the informed measure, prunedMeasure_ will be set to si_->getSpaceMeasure();
  r_rrt_ =
    rewireFactor_ *
    std::pow(2 * (1.0 + 1.0 / dimDbl) * (prunedMeasure_ / unitNBallMeasure(si_->getStateDimension())), 1.0 / dimDbl);
}

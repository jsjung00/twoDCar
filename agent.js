import { Car } from "./car";
import { Model } from "./model";
import { ReplayMemory } from "./replay_memory";

import * as tf from "@tensorflow/tfjs";
import { ALL_ACTIONS, NUM_ACTIONS, getRandomAction } from "./car_game";
import { generateTraffic } from "./utils";
import { animate } from "./main";
import { carCanvasWidth } from "./train";
import { Road } from "./road";
import { differenceBy } from "lodash";
const MIN_EPSILON = 0.01;
const MAX_EPSILON = 0.2;
const LAMBDA = 0.01;

export class CarGameAgent {
  constructor(car, hiddenLayerSizes, config) {
    this.car = car;
    const road = new Road(carCanvasWidth / 2, carCanvasWidth * 0.9, 3);
    this.road = road;
    this.hiddenLayerSizes = hiddenLayerSizes;

    this.epsilonInit = config.epsilonInit;
    this.epsilonFinal = config.epsilonFinal;
    this.epsilonDecayFrames = config.epsilonDecayFrames;
    this.epsilonIncrement_ =
      (this.epsilonFinal - this.epsilonInit) / this.epsilonDecayFrames;
    //networks
    this.onlineNetwork = new Model(
      hiddenLayerSizes,
      this.car.sensor.rayCount,
      NUM_ACTIONS
    );
    this.targetNetwork = new Model(
      hiddenLayerSizes,
      this.car.sensor.rayCount,
      NUM_ACTIONS
    );
    //freeze target network
    this.targetNetwork.network.trainable = false;
    this.optimizer = tf.train.adam(config.learningRate);
    this.replayBufferSize = config.replayBufferSize;
    this.replayMemory = new ReplayMemory(config.replayBufferSize);
    this.frameCount = 0;

    this.reset();
  }
  reset() {
    this.traffic = generateTraffic(30, 100, 200, this.road, -100, 0.3);
    this.car.resetLocation();
    //get first sensor readings
    this.car.sensor.update(this.road.borders, this.traffic);
    //reward info
    this.cumulativeReward_ = 0;
    this.numCheckPoints = 3000;
    this.passedCheckPoint = new Array(this.numCheckPoints).fill(false);
    this.furthestDistance = 0;
    this.maxPosReward = 1000;
  }

  playStep() {
    this.epsilon =
      this.frameCount >= this.epsilonDecayFrames
        ? this.epsilonFinal
        : this.epsilonInit + this.epsilonIncrement_ * this.frameCount;
    this.frameCount++;
    //epsilon-greedy
    let action;
    const state = this.car.getStateTensor();

    if (Math.random() < this.epsilon) {
      action = getRandomAction();
    } else {
      //greedily pick an action
      tf.tidy(() => {
        action =
          ALL_ACTIONS[
            this.onlineNetwork.network.predict(state).argMax(-1).dataSync()[0]
          ];
      });
    }

    //interact with environment, one game step
    /*for (let i = 0; i < this.traffic.length; i++) {
      this.traffic[i].update(this.road.borders, []);
    }*/
    let reward;
    const status = this.car.update(this.road.borders, this.traffic, action);
    const done = status === "DONE";
    const dead = status === "DEAD";
    if (done) {
      reward = 100;
    } else if (dead) {
      reward = -100;
    } else {
      reward = this.computeReward(
        this.car,
        this.passedCheckPoint,
        this.numCheckPoints
      );
    }

    const nextState = this.car.getStateTensor();
    //add experience to replay memory
    this.replayMemory.append([state, action, reward, done || dead, nextState]);
    this.cumulativeReward_ += reward;

    //update furthest distance
    if (this.car.initY - this.car.y > this.furthestDistance) {
      this.furthestDistance = this.car.initY - this.car.y;
    }
    //we keep track of whether it finishes the goal- if it crashes then we reset without logging cumlative reward in train.js
    const output = {
      //action,
      cumulativeReward: this.cumulativeReward_,
      done: done,
      dead: dead,
      furthestDistance: this.furthestDistance,
      carY: this.car.y,
      //carX: this.car.x,
      //speed: this.car.speed,
    };

    if (done || dead) {
      this.reset();
    }
    return output;
  }

  trainOnReplayBatch(batchSize, gamma, optimizer) {
    //Get batch of examples from replay buffer
    const batch = this.replayMemory.sample(batchSize);

    const lossFunction = () => {
      const stateArrays = batch.map((example) =>
        Array.from(example[0].dataSync())
      );
      const stateTensor = tf.tensor2d(stateArrays, [
        batchSize,
        this.car.sensor.rayCount,
      ]);
      const actionTensor = tf.tensor1d(
        batch.map((example) => example[1]),
        "int32"
      );
      let qs = this.onlineNetwork.network.apply(stateTensor, {
        training: true,
      });
      qs = qs.mul(tf.oneHot(actionTensor, NUM_ACTIONS));
      qs = qs.sum(-1);
      const rewardTensor = tf.tensor1d(batch.map((example) => example[2]));
      const nextStateArrays = batch.map((example) =>
        Array.from(example[4].dataSync())
      );
      const nextStateTensor = tf.tensor2d(nextStateArrays, [
        batchSize,
        this.car.sensor.rayCount,
      ]);
      const nextMaxQTensor = this.targetNetwork.network
        .predict(nextStateTensor)
        .max(-1);
      const doneMask = tf
        .scalar(1)
        .sub(tf.tensor1d(batch.map((example) => example[3])).asType("float32"));
      const targetQs = rewardTensor.add(
        nextMaxQTensor.mul(doneMask).mul(gamma)
      );
      const MSE = tf.losses.meanSquaredError(targetQs, qs).asScalar();
      return MSE;
    };

    const grads = tf.variableGrads(lossFunction);
    // Use the gradients to update the online DQN's weights.
    optimizer.applyGradients(grads.grads);
    tf.dispose(grads);
  }

  computeReward(car, passedCheckPoint, numCheckPoints) {
    let totReward = 0;
    const zoneLength = Math.abs((car.goalY - car.initY) / numCheckPoints);
    //zone of car, indexed from 0
    const currentZone = Math.floor((car.initY - car.y) / zoneLength);
    //car has passed finished line zone
    if (currentZone >= numCheckPoints) return;

    if (currentZone >= 1 && passedCheckPoint[currentZone - 1] === false) {
      totReward += (this.maxPosReward - 100) / numCheckPoints;
      passedCheckPoint[currentZone - 1] = true;
    }
    //penalize at each time step
    totReward -= 0.1;
    return totReward;
  }
}

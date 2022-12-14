import * as fs from "fs";

import * as argparse from "argparse";
import { mkdir } from "shelljs";

// The value of tf (TensorFlow.js-Node module) will be set dynamically
// depending on the value of the --gpu flag below.
let tf = require("@tensorflow/tfjs-node");

import { CarGameAgent } from "./agent";
import { copyWeights, Model } from "./model";
import { Car } from "./car";
import { NUM_ACTIONS } from "./car_game";
import { Road } from "./road";

export const carCanvasWidth = 300;

class MovingAverager {
  constructor(bufferLength) {
    this.buffer = [];
    for (let i = 0; i < bufferLength; ++i) {
      this.buffer.push(null);
    }
  }

  append(x) {
    this.buffer.shift();
    this.buffer.push(x);
  }

  average() {
    return this.buffer.reduce((x, prev) => x + prev) / this.buffer.length;
  }
}

export async function train(
  agent,
  batchSize,
  gamma,
  learningRate,
  cumulativeRewardThreshold,
  maxNumFrames,
  syncEveryFrames,
  savePath,
  logDir
) {
  let summaryWriter;
  if (logDir != null) {
    summaryWriter = tf.node.summaryFileWriter(logDir);
  }

  for (let i = 0; i < agent.replayBufferSize; ++i) {
    const output = agent.playStep();
    console.log(
      `Distance=${output.furthestDistance}, Epsilon=${agent.epsilon}`
    );
  }

  // Moving averager: cumulative reward across 100 most recent 100 episodes.
  const rewardAverager100 = new MovingAverager(100);
  // Moving averager: fruits eaten across 100 most recent 100 episodes.
  const distanceAverager100 = new MovingAverager(100);
  const optimizer = tf.train.adam(learningRate);
  let tPrev = new Date().getTime();
  let frameCountPrev = agent.frameCount;
  let averageReward100Best = -Infinity;
  let counter = 0;
  while (true) {
    counter++;
    agent.trainOnReplayBatch(batchSize, gamma, optimizer);
    const { cumulativeReward, done, furthestDistance } = agent.playStep();

    if (done) {
      const t = new Date().getTime();
      const framesPerSecond =
        ((agent.frameCount - frameCountPrev) / (t - tPrev)) * 1e3;
      tPrev = t;
      frameCountPrev = agent.frameCount;

      rewardAverager100.append(cumulativeReward);
      distanceAverager100.append(furthestDistance);
      const averageReward100 = rewardAverager100.average();
      const averageMaxDistance100 = distanceAverager100.average();

      console.log(
        `Frame #${agent.frameCount}: ` +
          `cumulativeReward100=${averageReward100.toFixed(1)}; ` +
          `distance100=${averageMaxDistance100.toFixed(2)} ` +
          `furthest=${furthestDistance}` +
          `(epsilon=${agent.epsilon.toFixed(3)}) ` +
          `(${framesPerSecond.toFixed(1)} frames/s)`
      );
      if (summaryWriter != null) {
        summaryWriter.scalar(
          "cumulativeReward100",
          averageReward100,
          agent.frameCount
        );
        summaryWriter.scalar(
          "distance100",
          averageMaxDistance100,
          agent.frameCount
        );
        summaryWriter.scalar("epsilon", agent.epsilon, agent.frameCount);
        summaryWriter.scalar(
          "framesPerSecond",
          framesPerSecond,
          agent.frameCount
        );
      }
      if (
        averageReward100 >= cumulativeRewardThreshold ||
        agent.frameCount >= maxNumFrames
      ) {
        // TODO(cais): Save online network.
        break;
      }
      if (averageReward100 > averageReward100Best) {
        averageReward100Best = averageReward100;
        if (savePath != null) {
          if (!fs.existsSync(savePath)) {
            mkdir("-p", savePath);
          }
          await agent.onlineNetwork.network.save(`file://${savePath}`);
          console.log(`Saved DQN to ${savePath}`);
        }
      }
    }
    if (agent.frameCount % syncEveryFrames === 0) {
      copyWeights(agent.targetNetwork.network, agent.onlineNetwork.network);
      console.log(
        "Sync'ed weights from online network to target network",
        counter
      );
    }
  }
}

export function parseArguments() {
  const parser = new argparse.ArgumentParser({
    description: "Training script for a DQN that plays the snake game",
  });
  parser.addArgument("--gpu", {
    action: "storeTrue",
    help:
      "Whether to use tfjs-node-gpu for training " +
      "(requires CUDA GPU, drivers, and libraries).",
  });
  parser.addArgument("--cumulativeRewardThreshold", {
    type: "float",
    defaultValue: 700,
    help:
      "Threshold for cumulative reward (its moving " +
      "average) over the 100 latest games. Training stops as soon as this " +
      "threshold is reached (or when --maxNumFrames is reached).",
  });
  parser.addArgument("--maxNumFrames", {
    type: "float",
    defaultValue: 1e6,
    help:
      "Maximum number of frames to run durnig the training. " +
      "Training ends immediately when this frame count is reached.",
  });
  parser.addArgument("--replayBufferSize", {
    type: "int",
    defaultValue: 1e5,
    help: "Length of the replay memory buffer.",
  });
  parser.addArgument("--epsilonInit", {
    type: "float",
    defaultValue: 0.5,
    help: "Initial value of epsilon, used for the epsilon-greedy algorithm.",
  });
  parser.addArgument("--epsilonFinal", {
    type: "float",
    defaultValue: 0.08,
    help: "Final value of epsilon, used for the epsilon-greedy algorithm.",
  });
  parser.addArgument("--epsilonDecayFrames", {
    type: "int",
    defaultValue: 2e5,
    help:
      "Number of frames of game over which the value of epsilon " +
      "decays from epsilonInit to epsilonFinal",
  });
  parser.addArgument("--batchSize", {
    type: "int",
    defaultValue: 64,
    help: "Batch size for DQN training.",
  });
  parser.addArgument("--gamma", {
    type: "float",
    defaultValue: 0.99,
    help: "Reward discount rate.",
  });
  parser.addArgument("--learningRate", {
    type: "float",
    defaultValue: 1e-3,
    help: "Learning rate for DQN training.",
  });
  parser.addArgument("--syncEveryFrames", {
    type: "int",
    defaultValue: 1e3,
    help:
      "Frequency at which weights are sync'ed from the online network " +
      "to the target network.",
  });
  parser.addArgument("--savePath", {
    type: "string",
    defaultValue: "./models/dqn",
    help: "File path to which the online DQN will be saved after training.",
  });
  parser.addArgument("--logDir", {
    type: "string",
    defaultValue: null,
    help: "Path to the directory for writing TensorBoard logs in.",
  });
  return parser.parseArgs();
}

async function main() {
  const args = parseArguments();
  if (args.gpu) {
    tf = require("@tensorflow/tfjs-node-gpu");
  }
  console.log(`args: ${JSON.stringify(args, null, 2)}`);
  //Canvas UI
  const road = new Road(carCanvasWidth / 2, carCanvasWidth * 0.9, 3);
  const drivingCar = new Car(road.getLaneCenter(1), -100, 50, 75, "AI", 3);
  console.log(
    `raycount ${drivingCar.sensor.rayCount} and num actions ${NUM_ACTIONS}`
  );
  const hiddenLayerSizes = [10];
  console.log("passed model");
  const agent = new CarGameAgent(drivingCar, hiddenLayerSizes, {
    replayBufferSize: args.replayBufferSize,
    epsilonInit: args.epsilonInit,
    epsilonFinal: args.epsilonFinal,
    epsilonDecayFrames: args.epsilonDecayFrames,
    learningRate: args.learningRate,
  });
  console.log("trying train");
  try {
    await train(
      agent,
      args.batchSize,
      args.gamma,
      args.learningRate,
      args.cumulativeRewardThreshold,
      args.maxNumFrames,
      args.syncEveryFrames,
      args.savePath,
      args.logDir
    );
  } catch (error) {
    console.error(error);
  }
}

if (require.main === module) {
  try {
    main();
  } catch (error) {
    console.error(error);
  }
}

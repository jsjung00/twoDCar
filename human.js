//human playing game to collect human memory

import * as tf from "./node_modules/@tensorflow/tfjs";
import {
  ACTION_DO_NOTHING,
  ACTION_GO_LEFT,
  ACTION_GO_RIGHT,
  ACTION_GO_STRAIGHT,
  ALL_ACTIONS,
  getRandomAction,
} from "./car_game";
import { animate } from "./main";
import { Car } from "./car";
import { Road } from "./road";
import { generateTraffic } from "./utils";
import { greedyNetwork } from "./utils";
const gameCanvas = document.getElementById("carCanvas");
gameCanvas.width = 300;
const gameCtx = gameCanvas.getContext("2d");
let cumulativeReward = 0;
let furthestDistance = 0;
const LOCAL_MODEL_URL = "./model/dqn/model.json";
const REMOTE_MODEL_URL = "https://jsjung00.github.io/car-model-json/model.json";

class GameScore {
  constructor(car, traffic, road) {
    this.car = car;
    this.traffic = traffic;
    this.road = road;
    this.cumulativeReward_ = 0;
    this.numCheckPoints = 3000;
    this.passedCheckPoint = new Array(this.numCheckPoints).fill(false);
    this.furthestDistance = 0;
  }
  reset() {
    this.car.resetLocation();
    this.cumulativeReward_ = 0;
    this.furthestDistance = 0;
    this.passedCheckPoint = new Array(this.numCheckPoints).fill(false);
    this.maxPosReward = 1000;
  }

  step(action) {
    //interact with environment, one game step
    /*for (let i = 0; i < this.traffic.length; i++) {
      this.traffic[i].update(this.road.borders, []);
    }*/
    let reward;
    const state = this.car.getStateArray();
    const status = this.car.update(this.road.borders, this.traffic, action);
    const done = status === "DONE";
    //const dead = status === "DEAD";
    const dead = this.car.damaged;
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
    const nextState = this.car.getStateArray();
    const distance = this.car.initY - this.car.y;
    return {
      state,
      action,
      reward,
      finished: Boolean(done || dead),
      nextState,
      distance,
    };
  }

  computeReward(car, passedCheckPoint, numCheckPoints) {
    let totReward = 0;
    const zoneLength = Math.abs((car.goalY - car.initY) / numCheckPoints);
    //zone of car, indexed from 0
    const currentZone = Math.floor((car.initY - car.y) / zoneLength);
    //car has passed finished line zone
    if (currentZone >= numCheckPoints) return 0;

    if (currentZone >= 1 && passedCheckPoint[currentZone - 1] === false) {
      totReward += (this.maxPosReward - 100) / numCheckPoints;
      passedCheckPoint[currentZone - 1] = true;
    }
    //penalize at each time step
    totReward -= 0.03;
    return totReward;
  }
}

let game;
let qNet;
let humanExperienceBuffer = [];
async function reset() {
  if (game === null) {
    return;
  }
  game.reset();
  animate(game.car, game.traffic, game.road, gameCanvas, gameCtx);
}

async function step() {
  let bestAction;
  if (!changedDirection) {
    bestAction = ACTION_DO_NOTHING;
  } else {
    changedDirection = false;
    switch (currentDirection) {
      case "LEFT":
        bestAction = ACTION_GO_LEFT;
        break;
      case "RIGHT":
        bestAction = ACTION_GO_RIGHT;
        break;
      case "STRAIGHT":
        bestAction = ACTION_GO_STRAIGHT;
    }
  }

  const { state, action, reward, finished, nextState, distance } =
    game.step(bestAction);
  console.log(`DISTANCE=${distance}`);
  //append to buffer
  cumulativeReward += reward;
  humanExperienceBuffer.push([state, action, reward, finished, nextState]);

  if (finished) {
    console.log("Game OVER\n\n\n\n\n");
    console.log(autoPlayIntervalJob);
    clearInterval(autoPlayIntervalJob);
    //save array in browser
    localStorage.setItem("humanBuffer", JSON.stringify(humanExperienceBuffer));
  }
  animate(game.car, game.traffic, game.road, gameCanvas, gameCtx);
}

let bestAction;
let changedDirection = false;
let currentDirection;
let autoPlayIntervalJob;

async function initGame() {
  console.log("gameCanvas width", gameCanvas.width);
  const road = new Road(gameCanvas.width / 2, gameCanvas.width * 0.9, 3);
  const drivingCar = new Car(road.getLaneCenter(0), -100, 50, 75, "HUMAN", 3);
  //TODO: add traffic cars
  const traffic = generateTraffic(50, 170, 250, road, -100, 0.3);

  //init car sensor
  drivingCar.sensor.update(road.borders, traffic);
  game = new GameScore(drivingCar, traffic, road);

  //warmup qNet
  for (let i = 0; i < 3; i++) {
    //qNet.predict(drivingCar.getStateTensor());
  }
  await reset();
  document.onkeydown = (event) => {
    switch (event.key) {
      case "ArrowLeft":
        console.log("left");
        currentDirection = "LEFT";
        changedDirection = true;
        break;
      case "ArrowRight":
        console.log("right");
        currentDirection = "RIGHT";
        changedDirection = true;
        break;
      case "ArrowUp":
        console.log("up");
        currentDirection = "STRAIGHT";
        changedDirection = true;
        break;
    }
  };
  autoPlayIntervalJob = setInterval(step, 10);
}

(async function () {
  try {
    initGame();
  } catch (err) {
    console.log("Loading failed");
    console.error(err);
  }
})();

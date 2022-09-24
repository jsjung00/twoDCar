import * as tf from "./node_modules/@tensorflow/tfjs";
import { ALL_ACTIONS, getRandomAction } from "./car_game";
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
    const state = this.car.getStateTensor();
    const distance = this.car.initY - this.car.y;
    return { reward, state, done, dead, distance };
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
    totReward -= 0.03;
    return totReward;
  }
}

let game;
let qNet;
async function reset() {
  if (game === null) {
    return;
  }
  game.reset();
  await calcQValuesAndBestAction();
  animate(game.car, game.traffic, game.road, gameCanvas, gameCtx);
}

async function step() {
  const { reward, done, dead, distance } = game.step(bestAction);
  invalidateQValuesAndBestAction();
  cumulativeReward += reward;
  if (dead) {
    console.log("Game OVER\n\n\n\n\n");
    console.log(autoPlayIntervalJob);
    clearInterval(autoPlayIntervalJob);
  }
  await calcQValuesAndBestAction();
  animate(game.car, game.traffic, game.road, gameCanvas, gameCtx);
}

let currentQValues;
let bestAction;
let epsilon = 0.1;
let autoPlayIntervalJob;
async function calcQValuesAndBestAction() {
  if (currentQValues != null) {
    return;
  }
  tf.tidy(() => {
    const states = Array.from(game.car.getStateTensor().dataSync());
    console.log("input states", states);
    bestAction = greedyNetwork(states);
    console.log(`best action ${bestAction}`);
  });
  /*
  if (Math.random() < epsilon) {
    bestAction = getRandomAction();
    return;
  }
  tf.tidy(() => {
    const stateTensor = game.car.getStateTensor();
    const predictOut = qNet.predict(stateTensor);
    currentQValues = predictOut.dataSync();
    bestAction = ALL_ACTIONS[predictOut.argMax(-1).dataSync()[0]];
  });*/
}

function invalidateQValuesAndBestAction() {
  currentQValues = null;
  bestAction = null;
}

async function initGame() {
  console.log("gameCanvas width", gameCanvas.width);
  const road = new Road(gameCanvas.width / 2, gameCanvas.width * 0.9, 3);
  const drivingCar = new Car(road.getLaneCenter(0), -100, 50, 75, "AI", 3);
  //TODO: add traffic cars
  const traffic = generateTraffic(100, 150, 350, road, -100, 0.3);
  //const traffic = [];
  //init car sensor
  drivingCar.sensor.update(road.borders, traffic);
  game = new GameScore(drivingCar, traffic, road);

  //warmup qNet
  for (let i = 0; i < 3; i++) {
    //qNet.predict(drivingCar.getStateTensor());
  }
  await reset();

  autoPlayIntervalJob = setInterval(() => {
    step();
  }, 5);
}

(async function () {
  try {
    //qNet = await tf.loadLayersModel(REMOTE_MODEL_URL);
    initGame();
  } catch (err) {
    console.log("Loading failed");
    console.error(err);
  }
})();

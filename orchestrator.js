import { Car } from "./car";
import { Model } from "./model";
import { Memory } from "./memory";

import * as tf from "@tensorflow/tfjs";
import { animate } from "./main";
const MIN_EPSILON = 0.01;
const MAX_EPSILON = 0.2;
const LAMBDA = 0.01;

export class Orchestrator {
  constructor(car, model, memory, discountRate, maxStepsPerGame) {
    this.car = car;
    this.traffic = [];
    this.model = model;
    this.memory = memory;
    this.eps = MAX_EPSILON;
    this.steps = 0;
    this.maxStepsPerGame = maxStepsPerGame;
    this.discountRate = discountRate;
    this.rewardStore = new Array();
    this.maxPositionStore = new Array();
    //game UI
    const carCanvas = document.getElementById("carCanvas");
    carCanvas.width = 200;
    this.carCanvas = carCanvas;
    const carCtx = carCanvas.getContext("2d");
    this.carCtx = carCtx;
    const road = new Road(carCanvas.width / 2, carCanvas.width * 0.9);
    this.road = road;
  }
  async run() {
    //init traffic
    this.traffic = [];
    this.car.resetLocation();
    let state = this.car.getStateTensor();
    let totalReward = 0;
    let numCheckPoints = 20;
    let passedCheckPoint = new Array(numCheckPoints).fill(false);

    let maxPosition = 100000;
    let step = 0;
    while (step < this.maxStepsPerGame) {
      //update traffic
      for (let i = 0; i < this.traffic.length; i++) {
        this.traffic[i].update(road.borders, []);
      }
      //render environment
      await animate(this.car, this.traffic, this.carCanvas, this.carCtx);

      //interact with environment
      const action = this.model.chooseAction(state, this.eps);
      const done = this.car.update(this.road.borders, this.traffic);
      const reward = this.computeReward(
        this.car,
        passedCheckPoint,
        numCheckPoints
      );

      let nextState = this.car.getStateTensor();

      //update max position
      if (this.car.y < maxPosition) {
        maxPosition = this.car.y;
      }
      if (done) nextState = null;

      this.memory.addSample([state, action, reward, nextState]);

      this.steps += 1;
      this.eps =
        MIN_EPSILON +
        (MAX_EPSILON - MIN_EPSILON) * Math.exp(-LAMBDA * this.steps);

      state = nextState;
      totalReward += reward;
      step += 1;
      //keep track of max position reached and store total reward
      if (done || step == this.maxStepsPerGame) {
        this.rewardStore.push(totalReward);
        this.maxPositionStore.push(maxPosition);
      }
    }
    await this.replay();
  }
  computeReward(car, passedCheckPoint, numCheckPoints) {
    let totReward = 0;
    const zoneLength = Math.abs((car.goalY - car.initY) / numCheckPoints);
    //zone of car, indexed from 0
    const currentZone = Math.abs(Math.floor((car.y - car.initY) / zoneLength));
    if (currentZone >= 1 && passedCheckPoint[currentZone - 1] === false) {
      totReward += 20;
      passedCheckPoint[currentZone - 1] = true;
    }
    //penalize at each time step
    totReward -= 1;
    return totReward;
  }
  async replay() {
    //sample from memory
    const batch = this.memory.sample(this.model.batchSize);
    const states = batch.map(([state, , ,]) => state);
    const nextStates = batch.map(([, , , nextState]) =>
      nextState ? nextState : tf.zeros([this.model.numStates])
    );
    //predict the values of each action at each state
    const qsa = states.map((state) => this.model.predict(state));
    //predict values of each action at each next state
    const qsad = nextStates.map((nextState) => this.model.predict(nextState));

    let x = new Array();
    let y = new Array();

    //update state rewards with discounted next state rewards
    batch.forEach(([state, action, reward, nextState], index) => {
      const currentQ = qsa[index];
      currentQ[action] = nextState
        ? reward + this.discountRate * qsad[index].max().dataSync()
        : reward;
      x.push(state.dataSync());
      y.push(currentQ.dataSync());
    });

    //Clean unused tensors
    qsa.forEach((state) => state.dispose());
    qsad.forEach((state) => state.dispose());

    //Reshape batches to be fed into network
    x = tf.tensor2d(x, [x.length, this.model.numStates]);
    y = tf.tensor2d(y, [y.length, this.model.numActions]);

    await this.model.train(x, y);

    x.dispose();
    y.dispose();
  }
}

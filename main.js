import * as tf from "@tensorflow/tfjs";

/*
const traffic = [
  new Car(road.getLaneCenter(1), -100, 30, 50, "DUMMY", 2),
  new Car(road.getLaneCenter(0), -300, 30, 50, "DUMMY", 2),
  new Car(road.getLaneCenter(2), -300, 30, 50, "DUMMY", 2),
];*/

export async function animate(car, traffic, carCanvas, carCtx) {
  carCanvas.height = window.innerHeight;
  carCtx.save();
  carCtx.translate(0, -car.y + carCanvas.height * 0.7);
  car.draw(carCtx, "blue", true);
  road.draw(carCtx);
  for (let i = 0; i < traffic.length; i++) {
    traffic[i].draw(carCtx, "red");
  }
  carCtx.restore();
  await tf.nextFrame();
}

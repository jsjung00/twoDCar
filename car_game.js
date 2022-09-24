import * as tf from "@tensorflow/tfjs";
import { assertPositiveInteger, getRandomInteger } from "./utils";

export const ACTION_GO_STRAIGHT = 0;
//export const ACTION_GO_BACK = 1;
export const ACTION_GO_RIGHT = 1;
export const ACTION_GO_LEFT = 2;
export const ALL_ACTIONS = [
  ACTION_GO_STRAIGHT,
  ACTION_GO_RIGHT,
  ACTION_GO_LEFT,
];
export const NUM_ACTIONS = ALL_ACTIONS.length;

export function getRandomAction() {
  return getRandomInteger(0, NUM_ACTIONS);
}

export interface Move {
  start: { row: number, col: number};
  end: { row: number, col: number };
  result: Array<Array<number>>;
}

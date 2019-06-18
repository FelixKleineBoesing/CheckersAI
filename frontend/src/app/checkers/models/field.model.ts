export interface Field {
  col: number;
  row: number;
  isHighlighted: boolean;
  stoneType: StoneType;
  hasWhiteBackground: boolean;
  id: number;
}

export enum StoneType {
  normalPl1 = 1,
  normalPl2 = -1,
  queenPl1 = 2,
  queenPl2 = -2,
  none = 0
}

import { Component, OnInit, Input, ViewChildren, QueryList } from '@angular/core';
import { Field, StoneType } from '../../models/field.model';
import { FieldComponent } from '../field/field.component';
import { HttpClient } from '@angular/common/http';
import { Move } from '../../models/move.model';

@Component({
  selector: 'app-board',
  templateUrl: './board.component.html',
  styleUrls: ['./board.component.sass']
})
export class BoardComponent implements OnInit {

  @Input() size: number;
  sizeArray: Array<number>;
  fields: Array<Array<Field>>;
  selectedField: Field;

  // Test
  moves: Array<Move>;
  moveIterator = 0;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.initBoardFields();
    this.initStones();

    // Test
    this.loadGame();
  }

  initBoardFields() {
    const fields: Array<Array<Field>> = Array.from(Array(this.size), () => Array(this.size));
    this.sizeArray = Array.from(Array(this.size).keys());

    this.sizeArray.forEach(row => {
      this.sizeArray.forEach( col => {
        const field: Field = {
          col,
          row,
          isHighlighted: false,
          stoneType: StoneType.none,
          hasWhiteBackground: ((row + col) % 2 === 0),
          id: (row * this.size) + col
        };
        fields[row][col] = field;
      });
    });
    this.fields = fields;
  }

  setStones(stones: Array<Array<number>>) {
    stones.forEach( (_el, row) => {
      _el.forEach((_el2, col) => {
        this.fields[row][col].stoneType = stones[row][col];
      });
    });
  }

  onFieldClicked(field: Field) {
    if (this.selectedField === field) {
      this.selectedField.isHighlighted = false;
      this.selectedField = undefined;
    } else if (this.selectedField === undefined) {
      this.selectedField = field;
      this.selectedField.isHighlighted = true;
    } else {
      this.selectedField.isHighlighted = false;
      this.selectedField = undefined;
    }
  }

  async moveStone(startField: Field, endField: Field) {
    return new Promise(async resolve => {
      const start = document.getElementById(`stone${startField.id}`);
      const end = document.getElementById(`stone${endField.id}`);
      const startOffsetTop = start.offsetTop;
      const startOffsetLeft = start.offsetLeft;
      const topDistance = end.offsetTop - start.offsetTop;
      const leftDistance = end.offsetLeft - start.offsetLeft;
      const maxDistance = Math.max(Math.abs(topDistance), Math.abs(leftDistance));
      start.style.position = 'absolute';

      for (let iteration = 1; iteration <= maxDistance; iteration++) {
        await this.timeout(1);
        start.style.top = startOffsetTop + Math.floor((topDistance / maxDistance) * iteration) + 'px';
        start.style.left = startOffsetLeft + Math.floor((leftDistance / maxDistance) * iteration) + 'px';
      }

      this.fields[endField.row][endField.col].stoneType = this.fields[startField.row][startField.col].stoneType;
      this.fields[startField.row][startField.col].stoneType = StoneType.none;
      start.style.top = startOffsetTop + 'px';
      start.style.left = startOffsetLeft + 'px';
      resolve();
    });
  }

  private timeout(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async performMove(move: Move) {
    await this.moveStone(this.fields[move.start.row][move.start.col],
      this.fields[move.end.row][move.end.col]);
    await this.timeout(0);
    this.setStones(move.result);
  }

  /// Testing
  loadGame() {
    this.http.get('assets/game.json').subscribe(res => {
      this.moves = res['game'];
    });
  }

  nextMove() {
    this.performMove(this.moves[this.moveIterator]);
    this.moveIterator++;
  }

  async autoMove() {
    do {
      await this.performMove(this.moves[this.moveIterator]);
      this.moveIterator++;
    } while (this.moveIterator < this.moves.length);
  }

  initStones() {
    this.moveIterator = 0;
    const stones = [
      [0, 1, 0, 1, 0, 1, 0, 1],
      [1, 0, 1, 0, 1, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0],
      [0, -1, 0, -1, 0, -1, 0, -1],
      [-1, 0, -1, 0, -1, 0, -1, 0]
    ];

    this.setStones(stones);
  }
}

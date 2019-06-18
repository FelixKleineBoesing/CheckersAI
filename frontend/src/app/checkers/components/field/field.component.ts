import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { Field, StoneType } from '../../models/field.model';

@Component({
  selector: 'app-field',
  templateUrl: './field.component.html',
  styleUrls: ['./field.component.sass']
})
export class FieldComponent implements OnInit {

  @Input() field: Field;
  @Output() clicked = new EventEmitter<Field>();

  constructor() {
  }

  ngOnInit(): void {
  }

  onFieldClick() {
    this.clicked.emit(this.field);
  }

}

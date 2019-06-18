import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IndexComponent } from './components/index/index.component';
import { BoardComponent } from './components/board/board.component';
import { FieldComponent } from './components/field/field.component';
import { FlexLayoutModule } from '@angular/flex-layout';
import { CheckersHttpService } from './services/checkers-http.service';
import { HttpClientModule } from '@angular/common/http';
@NgModule({
  declarations: [IndexComponent, BoardComponent, FieldComponent],
  imports: [
    CommonModule,
    FlexLayoutModule,
    HttpClientModule
  ],
  exports: [
    IndexComponent
  ],
  providers: [
    CheckersHttpService
  ]
})
export class CheckersModule { }

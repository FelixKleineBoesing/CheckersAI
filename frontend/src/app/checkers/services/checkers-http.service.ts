import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { environment } from 'src/environments/environment';
import { Observable } from 'rxjs';

@Injectable()
export class CheckersHttpService {

  constructor(private http: HttpClient) { }

  sendMove(start: { row: number, col: number }, end: { row: number, col: number }): Observable<any> {
    const params: HttpParams = new HttpParams().set('startRow', start.row.toString());
    params.append('startCol', start.col.toString());
    params.append('endCol', end.col.toString());
    params.append('endRow', end.row.toString());

    return this.http.get<any>(environment.url, { params });
  }

}

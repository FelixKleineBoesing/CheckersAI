import { TestBed } from '@angular/core/testing';

import { CheckersHttpService } from './checkers-http.service';

describe('CheckersHttpService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: CheckersHttpService = TestBed.get(CheckersHttpService);
    expect(service).toBeTruthy();
  });
});

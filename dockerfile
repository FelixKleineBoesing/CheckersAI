FROM python:3.7

COPY required.txt ./
RUN pip install --no-cache-dir -r required.txt

RUN touch is_docker

ENTRYPOINT ["python", "-u"]
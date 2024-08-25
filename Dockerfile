FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y make
RUN make install

CMD ["make", "format", "lint", "tests", "sonar"]

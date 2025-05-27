#include <Servo.h>
Servo doorservo;

#define DIR_X 2
#define STEP_X 3
#define EN_X 4

#define DIR_Y 5
#define STEP_Y 6
#define EN_Y 7

#define DIR_R 8
#define STEP_R 9
#define EN_R 10

#define STEP_DELAY 800     // µs for X/Y movement
#define R_STEP_DELAY 5000  // µs for rotation

bool ready_for_command = false;
String input = "";

void setup() {
  pinMode(DIR_X, OUTPUT);  pinMode(STEP_X, OUTPUT);  pinMode(EN_X, OUTPUT);
  pinMode(DIR_Y, OUTPUT);  pinMode(STEP_Y, OUTPUT);  pinMode(EN_Y, OUTPUT);
  pinMode(DIR_R, OUTPUT);  pinMode(STEP_R, OUTPUT);  pinMode(EN_R, OUTPUT);

  doorservo.attach(11);

  digitalWrite(EN_X, HIGH);
  digitalWrite(EN_Y, HIGH);
  digitalWrite(EN_R, HIGH);

  Serial.begin(9600);
  delay(2000);
  Serial.println("[Arduino] READY");
  ready_for_command = true;
}

void loop() {
  if (!ready_for_command) return;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      input.trim();
      if (input.length() >= 4) {
        ready_for_command = false;
        processCommand(input);
        input = "";

        while (Serial.available()) Serial.read();  // flush

        ready_for_command = true;
      } else {
        Serial.println("[Arduino] Invalid input (too short)");
        Serial.println("DONE");
        input = "";
      }
    } else {
      input += c;
    }
  }
}

void processCommand(String cmd) {
  if (cmd == "HOME") {
    Serial.println("[Arduino] HOMING triggered");
    Serial.println("DONE");
    return;
  }

  // 서보모터 명령 처리
  if (cmd == "SERVO_1") {
    doorservo.write(180);  // 문 열기
    Serial.println("[Arduino] Servo opened");
    Serial.println("DONE");
    return;
  }
  if (cmd == "SERVO_0") {
    doorservo.write(0);   // 문 닫기
    Serial.println("[Arduino] Servo closed");
    Serial.println("DONE");
    return;
  }

  int dix = -1, diy = -1, dirR = -1;
  long stx = 0, sty = 0, stR = 0;

  int i1 = cmd.indexOf(',');
  int i2 = cmd.indexOf(',', i1 + 1);
  int i3 = cmd.indexOf(',', i2 + 1);
  int i4 = cmd.indexOf(',', i3 + 1);
  int i5 = cmd.indexOf(',', i4 + 1);

  if (i5 > 0) {
    dix = cmd.substring(0, i1).toInt();
    diy = cmd.substring(i1 + 1, i2).toInt();
    stx = cmd.substring(i2 + 1, i3).toInt();
    sty = cmd.substring(i3 + 1, i4).toInt();
    dirR = cmd.substring(i4 + 1, i5).toInt();
    stR = cmd.substring(i5 + 1).toInt();

    Serial.print("[Arduino] Received: ");
    Serial.println(cmd);
    Serial.print("[Arduino] Parsed: Xdir="); Serial.print(dix);
    Serial.print(" Ydir="); Serial.print(diy);
    Serial.print(" Xstep="); Serial.print(stx);
    Serial.print(" Ystep="); Serial.print(sty);
    Serial.print(" Rdir="); Serial.print(dirR);
    Serial.print(" Rstep="); Serial.println(stR);

    moveMotor(DIR_X, STEP_X, EN_X, dix, stx);
    delay(100);
    moveMotor(DIR_Y, STEP_Y, EN_Y, diy, sty);
    delay(100);
    R_moveMotor(DIR_R, STEP_R, EN_R, dirR, stR);

    Serial.println("DONE");
  } else {
    Serial.print("[Arduino] Parse error: ");
    Serial.println(cmd);
    Serial.println("DONE");
  }
}

void moveMotor(int dirPin, int stepPin, int enPin, bool direction, long steps) {
  digitalWrite(enPin, LOW);
  digitalWrite(dirPin, direction);

  for (long i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(STEP_DELAY);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(STEP_DELAY);
  }

  digitalWrite(enPin, HIGH);
}

void R_moveMotor(int dirPin, int stepPin, int enPin, bool direction, long steps) {
  digitalWrite(enPin, LOW);
  digitalWrite(dirPin, direction);

  for (long i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(R_STEP_DELAY);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(R_STEP_DELAY);
  }

  digitalWrite(enPin, HIGH);
}

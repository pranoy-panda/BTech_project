#define START_CMD_CHAR '>'
#define END_CMD_CHAR '\n'
#define DIV_CMD_CHAR ','
#define PC_ACK 'a'
#define MEGA_ACK 'b'
#define CAM_READ_SIG 'c'

#define MAX_PWM 250
#define MIN_PWM 180

#include <SoftwareSerial.h>
SoftwareSerial mySerial(10, 11); // RX, TX

//Global variables declaration

//Motor A(left)
const int enablePinA = 6;
const int motorPin1  = 8;  // Pin 10 of L293
const int motorPin2  = 7;  // Pin 14 of L293

//Motor B(right)
const int enablePinB = 3;
const int motorPin3  = 5; // Pin  7 of L293
const int motorPin4  = 4;  // Pin  2 of L293

// variables for orientation data processing
float value0, value1, value2;
int firstLogCount = 64000;
float initial_pitch,initial_roll;
float yaw,roll,pitch;
int pwmL,pwmR,forw,turning;

// motor control functions
void forward(int pwm)
{   
  analogWrite(enablePinA,pwm);
  analogWrite(enablePinB,pwm);
  delay(500);
  digitalWrite(enablePinA, LOW);
  digitalWrite(enablePinB,LOW); 
}

void backward(int pwm)
{
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, HIGH);
  digitalWrite(motorPin3, LOW);
  digitalWrite(motorPin4, HIGH);
  
  analogWrite(enablePinA,pwm);
  analogWrite(enablePinB,pwm);
  delay(500);
  digitalWrite(enablePinA, LOW);
  digitalWrite(enablePinB,LOW);     

  digitalWrite(motorPin1, HIGH);
  digitalWrite(motorPin2, LOW);
  digitalWrite(motorPin3, HIGH);
  digitalWrite(motorPin4, LOW);
}
void right_turn(int pwm)
{
  // switch dir to right motor
  digitalWrite(motorPin3, LOW);
  digitalWrite(motorPin4, HIGH);
    
  analogWrite(enablePinA,pwm);
  analogWrite(enablePinB,pwm);
  delay(500);
  digitalWrite(enablePinA, LOW);
  digitalWrite(enablePinB,LOW); 

  // restoring normal config
  digitalWrite(motorPin3, HIGH);
  digitalWrite(motorPin4, LOW);  
}
void left_turn(int pwm)
{
  // switch dir to left motor
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, HIGH);
    
  analogWrite(enablePinA,pwm);
  analogWrite(enablePinB,pwm);
  delay(500);
  digitalWrite(enablePinA, LOW);
  digitalWrite(enablePinB,LOW); 

  // restoring normal config
  digitalWrite(motorPin1, HIGH);
  digitalWrite(motorPin2, LOW);   
}

void mixed_motion(int pwmL,int pwmR){
  analogWrite(enablePinA,pwmL);
  analogWrite(enablePinB,pwmR);
  delay(500);
  digitalWrite(enablePinA, LOW);
  digitalWrite(enablePinB,LOW); 
}

// orientation data processing functions
bool inNeighbourhood(float val,float init_val,int cycle_max,int cycle_min,int size_of_neigh){
  if(val-size_of_neigh<cycle_min){
    if(val<init_val+size_of_neigh && val>cycle_min){
      return 1;
    }
    else if(val<cycle_max && val>(cycle_max - (init_val-cycle_min))){
      return 1;
    }
    else
      return 0;
  }
  if(val+size_of_neigh>cycle_max){
    if(val<cycle_max && val>init_val-size_of_neigh)
      return 1;
    else if(val<(cycle_max-init_val) && val>cycle_min)
      return 1;
    else
      return 0;    
  }
  // normal case
  if(val<init_val+size_of_neigh && val>init_val-size_of_neigh)
    return 1;
  else
    return 0;  
}

bool isClockwiseAhead(float val, float init_val, int cycle_max, int cycle_min){
  // this function finds whether val lies ahead of init_val or not (w.r.t. clockwise orientation)
  if(init_val > cycle_max/2){
    if((val>init_val && val<360) || (val>0 && val<(init_val-cycle_max/2))){
      return 1;
    }
    else
      return 0;
  }
  else{
    if(val>init_val && val<(init_val+cycle_max/2))
      return 1;
    else
      return 0;  
  }
}
void setup() {
    Serial.begin(38400);
    mySerial.begin(38400);
    
    //Set pins as outputs
    pinMode(motorPin1, OUTPUT);
    pinMode(motorPin2, OUTPUT);
    pinMode(motorPin3, OUTPUT);
    pinMode(motorPin4, OUTPUT);
    pinMode(enablePinA,OUTPUT);
    pinMode(enablePinB,OUTPUT);

    digitalWrite(motorPin1, HIGH);
    digitalWrite(motorPin2, LOW);
    digitalWrite(enablePinA,LOW);
    digitalWrite(motorPin3, HIGH);
    digitalWrite(motorPin4, LOW);
    digitalWrite(enablePinB,LOW);
      
    Serial.flush();
}

void loop() {
  mySerial.flush();
  int inCommand = 0;
  int sensorType = 0;
  unsigned long logCount = 0L;

  char getChar = ' ';  //read serial
  char char_from_pc = ' ';  

  // wait for incoming data
  if (Serial.available() < 1 || mySerial.available() < 1) return; // if serial empty, return to loop().

  // parse incoming command from PC 
  char_from_pc = Serial.read();
  //if (char_from_pc == PC_ACK)
  //Serial.println(char_from_pc);
  if (char_from_pc != PC_ACK) return; // if no command start flag, return to loop().
  // parse msg from bluetooth module
  getChar = mySerial.read();
  while(getChar!=START_CMD_CHAR){
    getChar = mySerial.read();
  }
  // parse incoming pin# and value  
  sensorType = mySerial.parseInt(); // read sensor typr
  logCount = mySerial.parseInt();  // read total logged sensor readings
  
  // orientation data
  yaw = mySerial.parseFloat();  // 1st sensor value
  pitch = mySerial.parseFloat();  // 2rd sensor value if exists
  roll = mySerial.parseFloat();  // 3rd sensor value if exists

  if(firstLogCount > logCount){
    firstLogCount = logCount;
    initial_pitch = pitch;
    initial_roll = roll;
  }

  if(pitch<initial_pitch+10 && pitch>initial_pitch-10){
    if(roll<initial_roll+10 && roll>initial_roll-10){
      // do nothing
      pwmL = 0;
      pwmR = 0;   
      Serial.println(MEGA_ACK);
      Serial.print(pwmL);
      Serial.print(',');
      Serial.println(pwmR);    
      Serial.println(CAM_READ_SIG);      
    }
    else{
      forw = map(abs(roll-initial_roll), 0, 70, MIN_PWM, MAX_PWM);
      pwmL = forw;
      pwmR = forw;
            
      if(roll<initial_roll){
        // move forward
        //Serial.println("forward");
    
        // high level motor command
        Serial.println(MEGA_ACK);
        Serial.print(pwmL);
        Serial.print(',');
        Serial.println(pwmR);        
        forward(forw); 
        Serial.println(CAM_READ_SIG);      
      }
      else{
        // move backward
        //Serial.println("backward");

        // high level motor command
        //backward(forw);
      }
    }
  }
  else if(roll<initial_roll+10 && roll>initial_roll-10){
    if(pitch<initial_pitch+10 && pitch>initial_pitch-10){
      // do nothing
      pwmL = 0;
      pwmR = 0;
           
    }
    else{
      // program for only turning
      
      turning = map(abs(pitch - initial_pitch),0,45,MIN_PWM,MAX_PWM);
      if(pitch<initial_pitch){
        //Serial.print("only turning right");
        // right turn
        pwmL = turning;
        pwmR = 0;
           
        //high level motor command
        Serial.println(MEGA_ACK);
        Serial.print(pwmL);
        Serial.print(',');
        Serial.println(pwmR);        
        right_turn(pwmL);  
        Serial.println(CAM_READ_SIG);       
      }
      else{
        //left turn
        //Serial.print("only turning left");
        pwmL = 0;
        pwmR = turning;
                   
        //high level motor command
        Serial.println(MEGA_ACK);
        Serial.print(pwmL);
        Serial.print(',');
        Serial.println(pwmR);        
        left_turn(pwmR);     
        Serial.println(CAM_READ_SIG);    
      }      
    }
  } 
  else{ // both turining and moving forward
    //Serial.println("turning and moving forward");
    // map(value, fromLow, fromHigh, toLow, toHigh)
    forw = map(roll-initial_roll, -60, 60, MIN_PWM, MAX_PWM);
    turning = map(abs(pitch - initial_pitch),0,45,5,25);
    if(pitch<initial_pitch){// forward and right
      pwmL = forw + turning;
      pwmR = forw - turning;
    }
    else{
      pwmL = forw - turning;
      pwmR = forw + turning;      
    }
        
    // high level motor command
    Serial.println(MEGA_ACK);
    Serial.print(pwmL);
    Serial.print(',');
    Serial.println(pwmR);     
    mixed_motion(pwmL,pwmR); 
    Serial.println(CAM_READ_SIG);  
  }

  
}

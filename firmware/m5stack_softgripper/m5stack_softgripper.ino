// SoftTwin – M5Stack (ESP32) soft gripper controller, micro-ROS over Wi-Fi (UDP) to the micro-ROS agent.
// Same topics as core/virtual_m5stack.py, so the twin and the engine do not care whether the hardware is real:
//   sub  /softtwin/pressure_cmd  std_msgs/Float32  [kPa]
//   pub  /real/pressure          std_msgs/Float32  [kPa]  50 Hz
//   pub  /real/bend              std_msgs/Float32  [deg]  50 Hz (flex sensor or IMU on the fingertip)
// Build: Arduino IDE / PlatformIO + micro_ros_arduino (Jazzy). Agent on the PC:
//   pixi run --manifest-path C:\pai_ros\pixi.toml ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888
// TODO(hardware): set pins / sensor calibration below to match your build. Values are placeholders.
#include <M5Unified.h>
#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/float32.h>

// ---- wiring (placeholders) ----
const int PIN_PUMP = 26;       // MOSFET gate -> 12 V diaphragm pump
const int PIN_VALVE = 25;      // MOSFET gate -> 3/2 solenoid valve (open = vent)
const int PIN_PRESSURE = 36;   // analog pressure sensor output
const int PIN_FLEX = 35;       // flex sensor divider (bend)
const float P_GAIN_KPA_PER_V = 25.0f, P_OFFSET_V = 0.2f;   // sensor datasheet / calibration
const float FLEX_DEG_PER_V = 60.0f, FLEX_OFFSET_V = 1.1f;  // calibrate with the twin (learning/actuator_net.py)
const float P_MAX = 80.0f;     // safety limit [kPa]

char WIFI_SSID[] = "YOUR_SSID";
char WIFI_PASS[] = "YOUR_PASSWORD";
char AGENT_IP[] = "192.168.0.10";
const int AGENT_PORT = 8888;

rcl_publisher_t pub_p, pub_b;
rcl_subscription_t sub_cmd;
std_msgs__msg__Float32 msg_p, msg_b, msg_cmd;
rclc_executor_t executor;
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rcl_timer_t timer;
volatile float p_cmd = 0.0f;

float readVolt(int pin) { return analogReadMilliVolts(pin) / 1000.0f; }

void on_cmd(const void *m) { p_cmd = constrain(((const std_msgs__msg__Float32 *)m)->data, 0.0f, P_MAX); }

void on_timer(rcl_timer_t *, int64_t) {
  float p = (readVolt(PIN_PRESSURE) - P_OFFSET_V) * P_GAIN_KPA_PER_V;
  float b = (readVolt(PIN_FLEX) - FLEX_OFFSET_V) * FLEX_DEG_PER_V;
  // bang-bang pressure control with hysteresis (good enough for a first prototype)
  if (p > P_MAX) { digitalWrite(PIN_PUMP, LOW); digitalWrite(PIN_VALVE, HIGH); }
  else if (p < p_cmd - 1.5f) { digitalWrite(PIN_PUMP, HIGH); digitalWrite(PIN_VALVE, LOW); }
  else if (p > p_cmd + 1.5f) { digitalWrite(PIN_PUMP, LOW); digitalWrite(PIN_VALVE, HIGH); }
  else { digitalWrite(PIN_PUMP, LOW); digitalWrite(PIN_VALVE, LOW); }
  msg_p.data = p; msg_b.data = b;
  rcl_publish(&pub_p, &msg_p, NULL);
  rcl_publish(&pub_b, &msg_b, NULL);
  M5.Display.setCursor(0, 0);
  M5.Display.printf("cmd %5.1f kPa\np   %5.1f kPa\nbend %5.1f deg   ", p_cmd, p, b);
}

void setup() {
  M5.begin();
  pinMode(PIN_PUMP, OUTPUT); pinMode(PIN_VALVE, OUTPUT);
  digitalWrite(PIN_PUMP, LOW); digitalWrite(PIN_VALVE, HIGH);  // start vented (safe)
  M5.Display.setTextSize(2);
  set_microros_wifi_transports(WIFI_SSID, WIFI_PASS, AGENT_IP, AGENT_PORT);
  delay(2000);
  allocator = rcl_get_default_allocator();
  rclc_support_init(&support, 0, NULL, &allocator);
  rclc_node_init_default(&node, "m5stack_softgripper", "", &support);
  rclc_publisher_init_default(&pub_p, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "/real/pressure");
  rclc_publisher_init_default(&pub_b, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "/real/bend");
  rclc_subscription_init_default(&sub_cmd, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32), "/softtwin/pressure_cmd");
  rclc_timer_init_default(&timer, &support, RCL_MS_TO_NS(20), on_timer);
  rclc_executor_init(&executor, &support.context, 2, &allocator);
  rclc_executor_add_subscription(&executor, &sub_cmd, &msg_cmd, &on_cmd, ON_NEW_DATA);
  rclc_executor_add_timer(&executor, &timer);
}

void loop() {
  M5.update();
  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(5));
}

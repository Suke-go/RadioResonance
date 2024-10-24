# sudo apt-get install python3-pip
# pip3 install RPi.GPIO
# これをterminalで実行する必要がある．

import RPi.GPIO as GPIO
import time
from datetime import datetime, timedelta

class TimeDialController:
    def __init__(self):
        # 別の資料を参照．ブレッドボードでつなぐ．
        self.CLK = 17
        self.DT = 18
        self.SW = 27
        
        #1930年代から開始して，1940年に終わる．
        self.base_year = 1930
        self.end_year = 1945
        self.current_year = 1930
        self.current_month = 1
        
        # エンコーダーの状態管理
        self.last_counter = 0
        self.last_clk_state = None
        self.button_pressed = False
        
        # エンコーダの一回当たり
        self.months_per_step = 1
        
        self.setup_gpio()
        
    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.CLK, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.DT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.SW, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # 割り込み設定
        GPIO.add_event_detect(self.CLK, GPIO.BOTH, callback=self.rotate_change)
        GPIO.add_event_detect(self.SW, GPIO.FALLING, callback=self.button_callback, bouncetime=300)
        
        self.last_clk_state = GPIO.input(self.CLK)

    def rotate_change(self, channel):
        """回転検出時のコールバック"""
        try:
            clk_state = GPIO.input(self.CLK)
            dt_state = GPIO.input(self.DT)
            
            if clk_state != self.last_clk_state:
                if dt_state != clk_state:
                    # 時計回り
                    self.increment_time()
                else:
                    # 反時計回り
                    self.decrement_time()
                    
            self.last_clk_state = clk_state
            
        except Exception as e:
            print(f"Error in rotate_change: {e}")

    def increment_time(self):
        """時間を進める"""
        self.current_month += self.months_per_step
        if self.current_month > 12:
            self.current_month = 1
            self.current_year += 1
            
        if self.current_year > self.end_year:
            self.current_year = self.end_year
            self.current_month = 12
            
        self.time_changed()

    def decrement_time(self):
        """時間を戻す"""
        self.current_month -= self.months_per_step
        if self.current_month < 1:
            self.current_month = 12
            self.current_year -= 1
            
        if self.current_year < self.base_year:
            self.current_year = self.base_year
            self.current_month = 1
            
        self.time_changed()

    def time_changed(self):
        """時間が変更された時の処理"""
        print(f"Current Time: {self.current_year}/{self.current_month:02d}")


    def cleanup(self):
        GPIO.cleanup()

    def get_current_time(self):
        return {
            'year': self.current_year,
            'month': self.current_month
        }

if __name__ == "__main__":
    try:
        controller = TimeDialController()
        print("Time Dial Controller Started")
        print("Turn the encoder to change time")
        print("Press Ctrl+C to exit")
        print()
        
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        controller.cleanup()
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

"""
This script uses Selenium to automatically download royalty-free background music from Pixabay.
"""

def scrape_bgm(max_pages=850):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    for i in range(1, max_pages + 1):
        try: 
            url = f"https://pixabay.com/music/search/?order=ec&pagi={i}"
            driver.get(url)
            time.sleep(2)  
            
            download_buttons = driver.find_elements(By.CSS_SELECTOR, "button[aria-label='Download']")
            print(f"Page {i}: Found {len(download_buttons)} download buttons")
            
            if not download_buttons:
                break  
            
            for b in download_buttons:
                try:
                    b.click()
                    time.sleep(2)  
                    close_buttons = driver.find_elements(By.CSS_SELECTOR, "button.close, button[aria-label='Close']")
                    if close_buttons:
                        close_buttons[1].click()
                        time.sleep(1)  
                except Exception as e:
                    print(f"Error occurred: {e}")
                    continue
        except Exception as e:
            print(f'Error at page {i}: {e}')    
          
    driver.quit()
    print("Completed!")

if __name__ == "__main__":
    scrape_bgm()
import time
import emoji
import requests
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def lineNotify(token, msg):
        url = "https://notify-api.line.me/api/notify"
        headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        payload = {'message': msg}
        r = requests.post(url, headers=headers, params=payload)
        return r.status_code


#s = Service("D:\Daqi\Common\chromedriver.exe")
options = Options()
#不顯示瀏覽器
#options.add_argument("--headless")
#path_to_downloads = r'D:\Daqi\排程\wgrib2'
#prefs = {'profile.default_content_settings.popups': 0, 'download.default_directory': 'path_to_downloads'}
#options.add_experimental_option('prefs', prefs)
dr = webdriver.Chrome(executable_path = ChromeDriverManager().install(), options=options)
dr.get("https://web.pcc.gov.tw/prkms/tender/common/basic/indexTenderBasic")
#df = pd.DataFrame(columns=["機關名稱","標案案號名稱","傳輸次數","公告日期","截止投標","預算金額"])
gov_lst = ["經濟部水利署","行政院農業委員會農田水利署","行政院農業委員會水土保持局","行政院農業委員會林務局",
           "行政院農業委員會農糧署","行政院農業委員會農業試驗所","水利局","行政院環境保護署","環境保護局",
           "農業局","都市發展局","水務局","消防局","消防署","經濟發展局"]


dr.implicitly_wait(10)
#df_all=pd.DataFrame(columns=["項次","機關名稱","標案案號|名稱","傳輸次數","招標方式","採購性質","公告日期","截止投標","預算金額"])
df_all=pd.DataFrame()
for i in gov_lst:
    dr.find_element(By.ID,"orgName").clear()
    dr.find_element(By.ID,"orgName").send_keys(i)
    #勞務
    dr.find_element(By.XPATH,'//*[@id="basicTenderSearchForm"]/table/tbody/tr[7]/td[3]/label[3]').click()
    #送出查詢
    dr.find_element(By.XPATH,'//*[@id="basicTenderSearchForm"]/table/tbody/tr[11]/td/div/a').click()
    #print(i)
    dr.implicitly_wait(10)
    #跳頁後開始定位
    element = dr.find_element(By.XPATH,'//*[@id="tpam"]/tbody/tr')
    # 進一步定位到表格内容所在的td節點
    td_content = element.find_elements(By.TAG_NAME,"td")
    lst = []  # 存為list
    for td in td_content:
        lst.append(td.text)
    matrix=[]
    for j in range(0,len(lst)-1,9):
        matrix.append(lst[j:j+9])
    matrixN=np.array(matrix)#轉np.array型
    matrixT=matrixN.T#矩陣轉置（按實際需求，如果不需要可以不轉置）
    df_temp=pd.DataFrame()
    for k in range(0,len(matrixT)):
        df_temp['%s'%k]=matrixT[k]#將切分後的資料存入df中
    df_all = pd.concat([df_all,df_temp],axis=0)
    lst.clear()
    matrix.clear()
    dr.back()
dr.quit()
df_all.dropna(inplace=True)
#df_all.drop_duplicates(subset ="1", keep = 'first', inplace = True) 
df_all.columns= ["項次", "機關名稱", "標案案號|名稱","傳輸次數","招標方式", "採購性質", "公告日期","截止投標","預算金額"]
#刪除重複的，因為大單位會把小單位的列出來
df_all.drop_duplicates(subset=["標案案號|名稱"],keep='first',inplace=True)
#整理表
df_all['標案案號']=df_all['標案案號|名稱'].map(lambda x:x.split('\n')[0])
df_all['標案名稱']=df_all['標案案號|名稱'].map(lambda x:x.split('\n')[1])
df_final = df_all.drop(columns=["項次","標案案號|名稱"])
df_final.reset_index(drop=True, inplace = True)
order = ["機關名稱", "標案案號","標案名稱",
         "傳輸次數","招標方式", "採購性質", "公告日期","截止投標","預算金額"]
df_final = df_final[order]
# df_final

df_final.to_csv(r'D:\\tender\\tenders.csv', sep=",", encoding='utf_8_sig', index=None)
#print('ok')

#群組的
token = "KQNILlqD12vGyntCczF6fazw0ClP3Gkp7VRMyXONloZ"
#以日期設為開頭
#print(date)
if len(df_final.index)==0:
    msg = "今日中午前沒有標案"+emoji.emojize(':zzz:')
    lineNotify(token, msg)
else:
    for i in range(0,len(df_final.index)):
        if i==0:
            msg = str(df_all.iloc[0][6])[4:]+"\n"
        end_date = str(df_all.iloc[i][7])[4:]
        money = str(df_all.iloc[i][8])
        # 有些特殊的表情需要指定 use_aliases=True 
        if (i+1)%15!=0:
            msg = msg +emoji.emojize(':star:') +str(df_final["機關名稱"][i]).strip()+"\n"+emoji.emojize(':four_leaf_clover:')+str(df_final["標案名稱"][i])+"\n"+emoji.emojize('date:')+end_date+"截止\n"+emoji.emojize('money:')+ money+"\n\n"
            #print(msg)
        if (i+1)%15==0:
            lineNotify(token, msg)
            if i!=len(df_final.index)-1:
                msg = str(df_all.iloc[0][6])[4:]+"\n"
            else:
                break
            #最後一筆
        if i==len(df_final.index)-1:
            lineNotify(token, msg)
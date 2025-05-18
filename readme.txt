======================= HOW TO RUN API =======================
Này là một file note
Chạy app.py với lệnh:
    "python app.py"
    API này sẽ run ở port 43210
    2 thành phần là API KEY và server URL được để ở dưới nha!
API_KEY = "a6f58e8a1c3e61d6f02b6c7b56d7b255831bb7fa1f2447bd8f1ac9283f083d6b"
Server url: 142.126.31.91:50646

======================= HOW TO TEST API =======================
B1: Tải Postman về https://www.postman.com/downloads/ , cùng với file "/workspace/XAI API Test.postman_collection.json"
B2: Import file json kia vô và chạy cái POST, nếu có phản hồi là oke r đó (Nhớ run API trước khi chạy)

======================= HOW TO SETUP API =======================

Từ line 62 tới 92 ở app.py chỉ là hàm tượng trưng, tụi m chỉ cần def cái main.py rồi import vô app.py cho có 2 hàm INIT và RESPONSE là được

Hàm INIT -> Hàm khởi tạo toàn bộ module hoặc model, sẽ được khởi chạy 1 lần duy nhất (ở line 94)
Nhưng mà sử dụng Mistral và LLaMA thì t nghĩ load 2 model cùng lúc hơi khó á :v có thể để trong flow response, mỗi lần chạy là load xong unload lun

Hàm RESPONSE -> Input: questions, premise-nl + (model, module,... được load ở INIT) 
$ LƯU Ý $: Người ta sẽ chỉ request 1 cặp questions-premise mỗi lần call API, nên xử lí theo y chang input vào output của hàm dummy nha
Nhớ replace lại cái hàm đó ở line 163 
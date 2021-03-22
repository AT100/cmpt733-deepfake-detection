## How to run

* This server uses Flask. (`flask` & `werkzeug` need to be installed to run this server)

```python
#conda
conda install -c anaconda flask
conda install -c conda-forge werkzeug
```

* Type `python .\server.py` to run the server
* The server will run on `localhost:5000`  Press CTRL+C to quit.



## How to use

* `/video` is the page user can upload the video for detection
* After uploading it, It will auto redirect to `/result` page, which should show the uploaded video is `Fake` or `Real`



## Bugfix

* If you change the code, but the page doesn't change, it might be the old version still running on the port 5000.

```python
#Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F  #change <PID> to the PID you find from last command

#mac
kill $(lsof -ti:5000)
```


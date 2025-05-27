import time

def test_model_inference_time():
    model = MyModel()
    X_sample = np.random.rand(1, 10)
    
    start_time = time.time()
    model.predict(X_sample)
    elapsed_time = time.time() - start_time
    
    assert elapsed_time < 0.01  # 10ミリ秒以内ならOK

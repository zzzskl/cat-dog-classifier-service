from fastapi.testclient import TestClient
from app.main import app  # 从我们的应用中导入 FastAPI 实例

# 创建一个测试客户端
client = TestClient(app)

def test_health_check():
    """
    Tests if the /health endpoint returns a 200 OK status and correct body.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_model_info():
    """
    Tests if the /info endpoint returns a 200 OK status and the correct model info.
    """
    response = client.get("/info")
    assert response.status_code == 200
    # 检查响应体是否包含预期的键
    response_data = response.json()
    assert "project_name" in response_data
    assert "model_name" in response_data
    assert "model_alias" in response_data

def test_predict_endpoint_without_file():
    """
    Tests that the /predict endpoint returns an error if no file is provided.
    """
    response = client.post("/predict")
    # FastAPI's TestClient correctly identifies missing file and returns 422 Unprocessable Entity
    assert response.status_code == 422
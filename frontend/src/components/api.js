import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

export const analyzeFile = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  return axios.post(`${API_BASE_URL}/analyze`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

from menbench.server.base import VLMInferenceServer, AGENT
import os
import base64
import requests
import json

from menbench.data import Sample, Obs

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@AGENT.register("AgentApiServer")
class AgentApiServer(VLMInferenceServer):
    prompt_suffix = """
### Task:
Predict the next discrete action at Step {} immediately. 

### Constraints:
- DO NOT provide any reasoning, analysis, or extra text.
- ONLY output the final result in the exact format: [Action Output]: [Yaw, Longitudinal, Lateral]

[Action Output]:"""
    action_definition = """
Each action is a 3D vector [Yaw, Longitudinal, Lateral]. Values are integers from -2 to 2:
- Yaw (Rotation): 2 (Fast Left), 1 (Slow Left), 0 (Straight), -1 (Slow Right), -2 (Fast Right). [Positive=Left, Negative=Right]
- Longitudinal (Forward/Backward): 2 (Fast Forward), 1 (Slow Forward), 0 (Stop), -1 (Slow Backward), -2 (Fast Backward). [Positive=Forward, Negative=Backward]
- Lateral (Sideways): 2 (Fast Left-side-step), 1 (Slow Left-side-step), 0 (Stop), -1 (Slow Right-side-step), -2 (Fast Right-side-step). [Positive=Left, Negative=Right]
"""

    def __init__(self, config: dict):
        super().__init__(config=config)
    
    def run(self, sample:dict)->requests.models.Response:
        payload, headers = self.build_input(sample)
        return requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload), timeout=120)

    def build_input(self, sample:Sample)->dict:
        """
        Build payload and headers.
        """

        prompt = self._build_prompt(sample.obs)
        content = [{"type": "text", "text": prompt}]
        for img_path in sample.obs.images:
            full_path = os.path.join(self.config["image_root"], img_path)
            try:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(full_path)}"}})
            except: pass

        return {"model": self.config['model'], "messages": [{"role": "user", "content": content}], "max_tokens": 1024, "temperature": 0.0}, \
               {"Authorization": f"Bearer {self.config['api_key']}", "Content-Type": "application/json"}

    def _build_prompt(self, obs:Obs)->str:
        """
        Build prompt.
        obs: dict[
            'images': list[str, ...]
            'past_actions': list[list[int, ...], ...]
            'imu': list[list[float, ...], ...]
        ]
        """
        K = len(obs.images)
        imu_data_str = "".join([f"Step {i+1}: Orientation={imu[0:4]}, Gyro={imu[4:7]}, Accel={imu[7:10]}\n" for i, imu in enumerate(obs.imu)])
        past_actions_str = "".join([f"Action Step {i+1}: {act}\n" for i, act in enumerate(obs.past_actions)])

        prompt_prefix = f"You are an expert quadruped robot controller.\n{self.action_definition}\n### Input Observations:\n1. Visual Stream: {K} consecutive egocentric images.\n"
        
        if self.config["no_imu"]:
            input_obs = f"2. Past Control Sequence:\n{past_actions_str}\n"
        else:
            input_obs = f"2. IMU Sensors:\n{imu_data_str}\n3. Past Control Sequence:\n{past_actions_str}\n"

        return prompt_prefix + input_obs + self.prompt_suffix.format(K)

    def reset(self):
        pass
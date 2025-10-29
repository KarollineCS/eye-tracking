from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class HardwareConfig:
    camera_id: int = 0
    target_fps: int = 30
    resolution: Tuple[int, int] = (1920, 1080)
    confidence_threshold: float = 0.5

@dataclass
class AlgorithmConfig:
    face_detection_method: str = "dnn"
    iris_detection_method: str = "hybrid"
    use_mediapipe: bool = True
    use_pca_pose: bool = True
    use_eye_spheres: bool = True
    face_detection_interval: int = 3
    max_iris_movement: float = 200.0
    gaze_model_path: str = "models/p00.ckpt"

    # Parâmetros de sensibilidade do gaze
    gaze_focal_length: float = 400.0  # [DEPRECATED] Não usado mais com nova normalização
    gaze_gain: float = 1.0  # Fator de amplificação dos ângulos (ajustado para nova normalização)
    normalize_by_radius: bool = False  # [DEPRECATED] Agora normaliza por largura do olho automaticamente

@dataclass
class CalibrationConfig:
    method: str = "adaptive"
    min_points: int = 9
    max_points: int = 25
    samples_per_point: int = 25
    max_samples_per_point: int = 35
    quality_threshold: float = 0.8
    interval_enabled: bool = True
    interval_duration: float = 2.0
    countdown_duration: float = 3.0

@dataclass
class SituationalAwarenessConfig:
    attention_zones: Dict[str, List[Tuple[int, int, int, int]]] = None
    fixation_threshold: float = 100.0
    saccade_velocity_threshold: float = 30.0
    fatigue_blink_rate_threshold: float = 0.5
    attention_switch_threshold: float = 2.0
    
    def __post_init__(self):
        if self.attention_zones is None:
            self.attention_zones = {
                'instruments': [(50, 50, 200, 150)],
                'road': [(200, 100, 600, 400)],
                'mirrors': [(10, 10, 80, 60), (560, 10, 80, 60)],
                'center': [(300, 240, 100, 100)]
            }

@dataclass
class PerformanceConfig:
    max_processing_time: float = 0.033
    cache_size: int = 5
    cache_timeout: int = 3
    history_size: int = 2
    enable_multiprocessing: bool = False

@dataclass
class GazeTrackingConfig:
    hardware: HardwareConfig = None
    algorithm: AlgorithmConfig = None
    calibration: CalibrationConfig = None
    situational_awareness: SituationalAwarenessConfig = None
    performance: PerformanceConfig = None
    
    debug_enabled: bool = False
    collect_research_data: bool = True
    export_format: str = "csv"
    data_output_dir: str = "./data/output"
    
    def __post_init__(self):
        if self.hardware is None:
            self.hardware = HardwareConfig()
        if self.algorithm is None:
            self.algorithm = AlgorithmConfig()
        if self.calibration is None:
            self.calibration = CalibrationConfig()
        if self.situational_awareness is None:
            self.situational_awareness = SituationalAwarenessConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()

# Configuração padrão
DEFAULT_CONFIG = GazeTrackingConfig()

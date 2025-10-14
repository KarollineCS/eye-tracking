import time
from collections import deque
from typing import Dict, List, Optional

class PerformanceOptimizer:
    """Sistema de otimização de performance"""
    
    def __init__(self, config):
        self.config = config
        self.face_detection_interval = 3  # Detecta face a cada 3 frames
        self.frame_counter = 0
        self.last_known_faces = []
        self.face_tracking_active = False
        
        # Cache para resultados
        self.iris_cache = {}
        self.iris_cache_timeout = 5
        self.iris_cache_counter = 0
        
        # Controle de FPS
        self.target_fps = config.hardware.target_fps
        self.frame_time_budget = 1.0 / self.target_fps
        self.processing_times = deque(maxlen=30)
        
        print("⚡ Otimizador de Performance inicializado")
    
    def should_detect_face(self) -> bool:
        """Determina se deve executar detecção facial custosa"""
        self.frame_counter += 1
        
        # Se não tem faces conhecidas, detecta sempre
        if not self.last_known_faces:
            return True
            
        # Caso contrário, detecta a cada intervalo
        return self.frame_counter % self.face_detection_interval == 0
    
    def cache_iris_result(self, frame_id: str, iris_data: Dict):
        """Cache de resultados de detecção de íris"""
        self.iris_cache[frame_id] = {
            'data': iris_data,
            'timestamp': self.iris_cache_counter
        }
        self.iris_cache_counter += 1
        
        # Limpa cache antigo
        if len(self.iris_cache) > 10:
            oldest_key = min(self.iris_cache.keys(), 
                           key=lambda k: self.iris_cache[k]['timestamp'])
            del self.iris_cache[oldest_key]
    
    def get_cached_iris(self, frame_id: str) -> Optional[Dict]:
        """Recupera resultado de íris do cache se disponível"""
        if frame_id in self.iris_cache:
            cache_age = self.iris_cache_counter - self.iris_cache[frame_id]['timestamp']
            if cache_age <= self.iris_cache_timeout:
                return self.iris_cache[frame_id]['data']
        return None
    
    def track_processing_time(self, processing_time: float):
        """Monitora tempo de processamento"""
        self.processing_times.append(processing_time)
        
        # Ajusta intervalo de detecção baseado na performance
        avg_time = sum(self.processing_times) / len(self.processing_times)
        if avg_time > self.frame_time_budget * 0.8:
            self.face_detection_interval = min(5, self.face_detection_interval + 1)
        elif avg_time < self.frame_time_budget * 0.5:
            self.face_detection_interval = max(2, self.face_detection_interval - 1)
    
    def get_performance_stats(self) -> Dict:
        """Retorna estatísticas de performance"""
        if not self.processing_times:
            return {}
        
        times = list(self.processing_times)
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        # FPS estimado
        estimated_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'average_processing_time': avg_time,
            'max_processing_time': max_time,
            'min_processing_time': min_time,
            'estimated_fps': estimated_fps,
            'target_fps': self.target_fps,
            'face_detection_interval': self.face_detection_interval,
            'cache_size': len(self.iris_cache)
        }
    
    def is_performance_adequate(self) -> bool:
        """Verifica se a performance está adequada"""
        if not self.processing_times:
            return True
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        return avg_time <= self.frame_time_budget
    
    def suggest_optimizations(self) -> List[str]:
        """Sugere otimizações baseadas na performance atual"""
        suggestions = []
        stats = self.get_performance_stats()
        
        if stats.get('estimated_fps', 0) < self.target_fps * 0.8:
            suggestions.append("Reduzir resolução da câmera")
            suggestions.append("Aumentar intervalo de detecção facial")
            suggestions.append("Desabilitar visualizações desnecessárias")
        
        if self.face_detection_interval >= 5:
            suggestions.append("Performance baixa - considere hardware mais potente")
        
        if len(suggestions) == 0:
            suggestions.append("Performance adequada")
        
        return suggestions


class MemoryManager:
    """Gerenciador de memória para evitar vazamentos"""
    
    def __init__(self, max_cache_size: int = 100):
        self.max_cache_size = max_cache_size
        self.frame_cache = deque(maxlen=max_cache_size)
        self.result_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def cache_frame_result(self, frame_hash: str, result: Dict):
        """Armazena resultado de processamento de frame"""
        if len(self.result_cache) >= self.max_cache_size:
            # Remove item mais antigo
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        self.result_cache[frame_hash] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def get_cached_result(self, frame_hash: str, max_age: float = 0.1) -> Optional[Dict]:
        """Recupera resultado em cache se ainda válido"""
        if frame_hash in self.result_cache:
            cached = self.result_cache[frame_hash]
            if time.time() - cached['timestamp'] <= max_age:
                self.cache_hits += 1
                return cached['result']
        
        self.cache_misses += 1
        return None
    
    def get_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.result_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    def clear_cache(self):
        """Limpa todo o cache"""
        self.result_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class AdaptiveProcessor:
    """Processador adaptativo que ajusta qualidade baseado na performance"""
    
    def __init__(self, config):
        self.config = config
        self.current_quality = 'high'
        self.quality_levels = {
            'high': {'scale': 1.0, 'detection_interval': 1},
            'medium': {'scale': 0.75, 'detection_interval': 2},
            'low': {'scale': 0.5, 'detection_interval': 3}
        }
        self.performance_history = deque(maxlen=10)
        self.adjustment_cooldown = 5.0  # segundos
        self.last_adjustment = 0
    
    def update_performance(self, processing_time: float):
        """Atualiza histórico de performance"""
        self.performance_history.append(processing_time)
        
        # Ajusta qualidade se necessário
        if time.time() - self.last_adjustment > self.adjustment_cooldown:
            self.adjust_quality()
    
    def adjust_quality(self):
        """Ajusta nível de qualidade baseado na performance"""
        if len(self.performance_history) < 5:
            return
        
        avg_time = sum(self.performance_history) / len(self.performance_history)
        target_time = 1.0 / self.config.hardware.target_fps
        
        if avg_time > target_time * 1.5 and self.current_quality != 'low':
            # Performance ruim, reduz qualidade
            if self.current_quality == 'high':
                self.current_quality = 'medium'
            elif self.current_quality == 'medium':
                self.current_quality = 'low'
            
            self.last_adjustment = time.time()
            print(f"⬇️ Qualidade reduzida para {self.current_quality}")
            
        elif avg_time < target_time * 0.7 and self.current_quality != 'high':
            # Performance boa, aumenta qualidade
            if self.current_quality == 'low':
                self.current_quality = 'medium'
            elif self.current_quality == 'medium':
                self.current_quality = 'high'
            
            self.last_adjustment = time.time()
            print(f"⬆️ Qualidade aumentada para {self.current_quality}")
    
    def get_current_settings(self) -> Dict:
        """Retorna configurações atuais baseadas na qualidade"""
        return self.quality_levels[self.current_quality].copy()
    
    def should_process_frame(self, frame_counter: int) -> bool:
        """Determina se deve processar o frame atual"""
        settings = self.get_current_settings()
        return frame_counter % settings['detection_interval'] == 0
    
    def get_processing_scale(self) -> float:
        """Retorna fator de escala para processamento"""
        return self.quality_levels[self.current_quality]['scale']

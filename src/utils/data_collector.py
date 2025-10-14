import json
import csv
import time
import os
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Any
import numpy as np

class ResearchDataCollector:
    """Coletor de dados para pesquisa científica"""
    
    def __init__(self, config):
        self.config = config
        self.session_id = self.generate_session_id()
        self.participant_id = None
        self.scenario_type = None
        
        # Estrutura de dados da sessão
        self.session_data = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': time.time(),
                'end_time': None,
                'duration': None,
                'participant_id': self.participant_id,
                'scenario_type': self.scenario_type,
                'system_config': self.config_to_dict()
            },
            'calibration_data': [],
            'gaze_data': [],
            'sa_analysis': [],
            'performance_metrics': [],
            'events': []
        }
        
        # Buffers para coleta em tempo real
        self.gaze_buffer = deque(maxlen=1000)
        self.sa_buffer = deque(maxlen=500)
        self.event_buffer = deque(maxlen=100)
        
        # Contadores e estatísticas
        self.frame_counter = 0
        self.total_fixations = 0
        self.total_saccades = 0
        
        # Configurações de export
        self.output_dir = config.data_output_dir
        self.export_format = config.export_format
        
        # Cria diretório de saída
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"📊 Coletor de Dados inicializado - Sessão: {self.session_id}")
    
    def generate_session_id(self) -> str:
        """Gera ID único para a sessão"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def set_participant_info(self, participant_id: str, scenario_type: str = None):
        """Define informações do participante"""
        self.participant_id = participant_id
        self.scenario_type = scenario_type or 'default'
        
        self.session_data['session_info']['participant_id'] = participant_id
        self.session_data['session_info']['scenario_type'] = scenario_type
        
        print(f"👤 Participante: {participant_id}, Cenário: {scenario_type}")
    
    def config_to_dict(self) -> Dict:
        """Converte configuração para dicionário serializable"""
        try:
            # Extrai informações principais da configuração
            return {
                'hardware': {
                    'camera_id': self.config.hardware.camera_id,
                    'target_fps': self.config.hardware.target_fps,
                    'resolution': self.config.hardware.resolution,
                    'confidence_threshold': self.config.hardware.confidence_threshold
                },
                'algorithm': {
                    'face_detection_method': self.config.algorithm.face_detection_method,
                    'iris_detection_method': self.config.algorithm.iris_detection_method,
                    'face_detection_interval': self.config.algorithm.face_detection_interval
                },
                'calibration': {
                    'method': self.config.calibration.method,
                    'min_points': self.config.calibration.min_points,
                    'max_points': self.config.calibration.max_points,
                    'samples_per_point': self.config.calibration.samples_per_point
                }
            }
        except Exception as e:
            print(f"⚠️ Erro ao converter config: {e}")
            return {}
    
    def collect_calibration_data(self, calibration_points: List[Dict]):
        """Coleta dados de calibração"""
        timestamp = time.time()
        
        calibration_entry = {
            'timestamp': timestamp,
            'method': self.config.calibration.method,
            'total_points': len(calibration_points),
            'points': calibration_points,
            'quality_metrics': self.calculate_calibration_quality(calibration_points)
        }
        
        self.session_data['calibration_data'].append(calibration_entry)
        print(f"📐 Dados de calibração coletados: {len(calibration_points)} pontos")
    
    def calculate_calibration_quality(self, calibration_points: List[Dict]) -> Dict:
        """Calcula métricas de qualidade da calibração"""
        if not calibration_points:
            return {}
        
        try:
            # Extrai valores de qualidade se disponíveis
            qualities = [p.get('quality', 0) for p in calibration_points if 'quality' in p]
            
            if qualities:
                return {
                    'average_quality': np.mean(qualities),
                    'min_quality': np.min(qualities),
                    'max_quality': np.max(qualities),
                    'std_quality': np.std(qualities),
                    'points_above_threshold': sum(1 for q in qualities if q > 0.75)
                }
        except Exception as e:
            print(f"⚠️ Erro no cálculo de qualidade: {e}")
        
        return {}
    
    def collect_frame_data(self, frame_data: Dict):
        """Coleta dados de um frame processado"""
        self.frame_counter += 1
        timestamp = frame_data.get('timestamp', time.time())
        
        # Dados básicos do frame
        gaze_entry = {
            'frame_id': self.frame_counter,
            'timestamp': timestamp,
            'gaze_vectors': self.serialize_gaze_vectors(frame_data.get('gaze_vectors', {})),
            'screen_point': frame_data.get('screen_point'),
            'face_confidence': frame_data.get('face_confidence'),
            'processing_time': frame_data.get('processing_time', 0)
        }
        
        # Adiciona ao buffer
        self.gaze_buffer.append(gaze_entry)
        
        # Coleta análise de consciência situacional
        sa_analysis = frame_data.get('sa_analysis', {})
        if sa_analysis:
            self.collect_sa_analysis(sa_analysis, timestamp)
        
        # Detecta eventos importantes
        self.detect_events(frame_data, timestamp)
    
    def serialize_gaze_vectors(self, gaze_vectors: Dict) -> Dict:
        """Serializa vetores de gaze para JSON"""
        serialized = {}
        
        for eye_side, data in gaze_vectors.items():
            if isinstance(data, dict):
                serialized[eye_side] = {
                    'yaw': float(data.get('yaw', 0)),
                    'pitch': float(data.get('pitch', 0)),
                    'eye_center': [int(x) for x in data.get('eye_center', [0, 0])],
                    'iris_center': [int(x) for x in data.get('iris_center', [0, 0])]
                }
        
        return serialized
    
    def collect_sa_analysis(self, sa_analysis: Dict, timestamp: float):
        """Coleta análise de consciência situacional"""
        sa_entry = {
            'timestamp': timestamp,
            'frame_id': self.frame_counter,
            'overall_score': sa_analysis.get('situational_awareness_score', {}).get('overall_score', 0),
            'classification': sa_analysis.get('situational_awareness_score', {}).get('classification', 'unknown'),
            'attention_zone': sa_analysis.get('attention_distribution', {}).get('current_zone', 'unknown'),
            'zone_confidence': sa_analysis.get('attention_distribution', {}).get('zone_confidence', 0),
            'fatigue_level': sa_analysis.get('fatigue_indicators', {}).get('fatigue_level', 'unknown'),
            'fatigue_score': sa_analysis.get('fatigue_indicators', {}).get('fatigue_score', 0),
            'is_fixating': sa_analysis.get('fixation_patterns', {}).get('is_fixating', False),
            'fixation_duration': sa_analysis.get('fixation_patterns', {}).get('fixation_duration', 0),
            'is_saccade': sa_analysis.get('saccade_patterns', {}).get('is_saccade', False),
            'saccade_frequency': sa_analysis.get('saccade_patterns', {}).get('saccade_frequency', 0)
        }
        
        self.sa_buffer.append(sa_entry)
    
    def detect_events(self, frame_data: Dict, timestamp: float):
        """Detecta e registra eventos importantes"""
        events = []
        
        sa_analysis = frame_data.get('sa_analysis', {})
        
        # Eventos de fadiga
        fatigue_level = sa_analysis.get('fatigue_indicators', {}).get('fatigue_level', 'unknown')
        if fatigue_level == 'high':
            events.append({
                'type': 'high_fatigue',
                'timestamp': timestamp,
                'frame_id': self.frame_counter,
                'details': {'fatigue_level': fatigue_level}
            })
        
        # Eventos de baixa consciência situacional
        sa_score = sa_analysis.get('situational_awareness_score', {}).get('overall_score', 1.0)
        if sa_score < 0.3:
            events.append({
                'type': 'low_situational_awareness',
                'timestamp': timestamp,
                'frame_id': self.frame_counter,
                'details': {'sa_score': sa_score}
            })
        
        # Eventos de atenção fora das zonas críticas
        current_zone = sa_analysis.get('attention_distribution', {}).get('current_zone', '')
        if current_zone == 'out_of_zone':
            events.append({
                'type': 'attention_out_of_zone',
                'timestamp': timestamp,
                'frame_id': self.frame_counter,
                'details': {'zone': current_zone}
            })
        
        # Adiciona eventos ao buffer
        for event in events:
            self.event_buffer.append(event)
    
    def add_custom_event(self, event_type: str, details: Dict = None):
        """Adiciona evento personalizado"""
        event = {
            'type': event_type,
            'timestamp': time.time(),
            'frame_id': self.frame_counter,
            'details': details or {}
        }
        
        self.event_buffer.append(event)
        print(f"📝 Evento registrado: {event_type}")
    
    def save_session_data(self):
        """Salva todos os dados da sessão"""
        # Finaliza sessão
        end_time = time.time()
        self.session_data['session_info']['end_time'] = end_time
        self.session_data['session_info']['duration'] = end_time - self.session_data['session_info']['start_time']
        
        # Move buffers para dados principais
        self.session_data['gaze_data'] = list(self.gaze_buffer)
        self.session_data['sa_analysis'] = list(self.sa_buffer)
        self.session_data['events'] = list(self.event_buffer)
        
        # Adiciona estatísticas da sessão
        self.session_data['session_statistics'] = self.calculate_session_statistics()
        
        # Salva em diferentes formatos
        if self.export_format == 'json' or self.export_format == 'both':
            self.save_as_json()
        
        if self.export_format == 'csv' or self.export_format == 'both':
            self.save_as_csv()
        
        print(f"💾 Dados da sessão salvos: {self.session_id}")
    
    def calculate_session_statistics(self) -> Dict:
        """Calcula estatísticas da sessão"""
        try:
            gaze_data = list(self.gaze_buffer)
            sa_data = list(self.sa_buffer)
            events = list(self.event_buffer)
            
            # Estatísticas básicas
            stats = {
                'total_frames': len(gaze_data),
                'total_sa_analyses': len(sa_data),
                'total_events': len(events),
                'duration_seconds': self.session_data['session_info']['duration']
            }
            
            # Estatísticas de consciência situacional
            if sa_data:
                sa_scores = [d['overall_score'] for d in sa_data if 'overall_score' in d]
                if sa_scores:
                    stats['sa_statistics'] = {
                        'average_sa_score': np.mean(sa_scores),
                        'min_sa_score': np.min(sa_scores),
                        'max_sa_score': np.max(sa_scores),
                        'std_sa_score': np.std(sa_scores)
                    }
                
                # Distribuição de zonas de atenção
                zones = [d['attention_zone'] for d in sa_data if 'attention_zone' in d]
                zone_counts = {}
                for zone in zones:
                    zone_counts[zone] = zone_counts.get(zone, 0) + 1
                
                stats['attention_zone_distribution'] = zone_counts
                
                # Contagem de eventos por tipo
                event_counts = {}
                for event in events:
                    event_type = event.get('type', 'unknown')
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
                stats['event_distribution'] = event_counts
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Erro no cálculo de estatísticas: {e}")
            return {}
    
    def save_as_json(self):
        """Salva dados em formato JSON"""
        try:
            filename = f"{self.session_id}_complete.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
            
            print(f"📄 JSON salvo: {filename}")
            
        except Exception as e:
            print(f"❌ Erro ao salvar JSON: {e}")
    
    def save_as_csv(self):
        """Salva dados em formato CSV (múltiplos arquivos)"""
        try:
            # CSV dos dados de gaze
            self.save_gaze_csv()
            
            # CSV da análise de SA
            self.save_sa_csv()
            
            # CSV dos eventos
            self.save_events_csv()
            
            # CSV das estatísticas
            self.save_statistics_csv()
            
        except Exception as e:
            print(f"❌ Erro ao salvar CSVs: {e}")
    
    def save_gaze_csv(self):
        """Salva dados de gaze em CSV"""
        filename = f"{self.session_id}_gaze_data.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Cabeçalho
            header = [
                'frame_id', 'timestamp', 'screen_x', 'screen_y',
                'left_yaw', 'left_pitch', 'left_eye_center_x', 'left_eye_center_y',
                'right_yaw', 'right_pitch', 'right_eye_center_x', 'right_eye_center_y',
                'face_confidence', 'processing_time'
            ]
            writer.writerow(header)
            
            # Dados
            for entry in self.gaze_buffer:
                gaze_vectors = entry.get('gaze_vectors', {})
                screen_point = entry.get('screen_point', [None, None])
                
                left_gaze = gaze_vectors.get('left', {})
                right_gaze = gaze_vectors.get('right', {})
                
                row = [
                    entry.get('frame_id', ''),
                    entry.get('timestamp', ''),
                    screen_point[0] if screen_point and len(screen_point) >= 2 else '',
                    screen_point[1] if screen_point and len(screen_point) >= 2 else '',
                    left_gaze.get('yaw', ''),
                    left_gaze.get('pitch', ''),
                    left_gaze.get('eye_center', [None, None])[0] if left_gaze.get('eye_center') else '',
                    left_gaze.get('eye_center', [None, None])[1] if left_gaze.get('eye_center') else '',
                    right_gaze.get('yaw', ''),
                    right_gaze.get('pitch', ''),
                    right_gaze.get('eye_center', [None, None])[0] if right_gaze.get('eye_center') else '',
                    right_gaze.get('eye_center', [None, None])[1] if right_gaze.get('eye_center') else '',
                    entry.get('face_confidence', ''),
                    entry.get('processing_time', '')
                ]
                
                writer.writerow(row)
        
        print(f"📊 CSV de gaze salvo: {filename}")
    
    def save_sa_csv(self):
        """Salva análise de consciência situacional em CSV"""
        filename = f"{self.session_id}_sa_analysis.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Cabeçalho
            header = [
                'timestamp', 'frame_id', 'overall_score', 'classification',
                'attention_zone', 'zone_confidence', 'fatigue_level', 'fatigue_score',
                'is_fixating', 'fixation_duration', 'is_saccade', 'saccade_frequency'
            ]
            writer.writerow(header)
            
            # Dados
            for entry in self.sa_buffer:
                row = [
                    entry.get('timestamp', ''),
                    entry.get('frame_id', ''),
                    entry.get('overall_score', ''),
                    entry.get('classification', ''),
                    entry.get('attention_zone', ''),
                    entry.get('zone_confidence', ''),
                    entry.get('fatigue_level', ''),
                    entry.get('fatigue_score', ''),
                    entry.get('is_fixating', ''),
                    entry.get('fixation_duration', ''),
                    entry.get('is_saccade', ''),
                    entry.get('saccade_frequency', '')
                ]
                writer.writerow(row)
        
        print(f"🧠 CSV de SA salvo: {filename}")
    
    def save_events_csv(self):
        """Salva eventos em CSV"""
        filename = f"{self.session_id}_events.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Cabeçalho
            header = ['timestamp', 'frame_id', 'event_type', 'details']
            writer.writerow(header)
            
            # Dados
            for event in self.event_buffer:
                row = [
                    event.get('timestamp', ''),
                    event.get('frame_id', ''),
                    event.get('type', ''),
                    json.dumps(event.get('details', {}))
                ]
                writer.writerow(row)
        
        print(f"📋 CSV de eventos salvo: {filename}")
    
    def save_statistics_csv(self):
        """Salva estatísticas em CSV"""
        filename = f"{self.session_id}_statistics.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        stats = self.session_data.get('session_statistics', {})
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Cabeçalho
            header = ['metric', 'value']
            writer.writerow(header)
            
            # Estatísticas básicas
            basic_stats = [
                ('session_id', self.session_id),
                ('participant_id', self.participant_id),
                ('scenario_type', self.scenario_type),
                ('duration_seconds', stats.get('duration_seconds', '')),
                ('total_frames', stats.get('total_frames', '')),
                ('total_sa_analyses', stats.get('total_sa_analyses', '')),
                ('total_events', stats.get('total_events', ''))
            ]
            
            for metric, value in basic_stats:
                writer.writerow([metric, value])
            
            # Estatísticas de SA
            sa_stats = stats.get('sa_statistics', {})
            for metric, value in sa_stats.items():
                writer.writerow([f'sa_{metric}', value])
            
            # Distribuição de zonas
            zone_dist = stats.get('attention_zone_distribution', {})
            for zone, count in zone_dist.items():
                writer.writerow([f'zone_{zone}_count', count])
            
            # Distribuição de eventos
            event_dist = stats.get('event_distribution', {})
            for event_type, count in event_dist.items():
                writer.writerow([f'event_{event_type}_count', count])
        
        print(f"📈 CSV de estatísticas salvo: {filename}")
    
    def export_for_analysis(self, analysis_type='spss'):
        """Exporta dados em formato específico para análise estatística"""
        try:
            if analysis_type.lower() == 'spss':
                self.export_spss_format()
            elif analysis_type.lower() == 'r':
                self.export_r_format()
            elif analysis_type.lower() == 'matlab':
                self.export_matlab_format()
            else:
                print(f"⚠️ Formato de análise não suportado: {analysis_type}")
                
        except Exception as e:
            print(f"❌ Erro na exportação para análise: {e}")
    
    def export_spss_format(self):
        """Exporta dados em formato compatível com SPSS"""
        filename = f"{self.session_id}_spss_format.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Combina dados de gaze e SA em uma única tabela
        combined_data = []
        
        # Mapeia dados de SA por frame_id
        sa_by_frame = {sa['frame_id']: sa for sa in self.sa_buffer if 'frame_id' in sa}
        
        for gaze_entry in self.gaze_buffer:
            frame_id = gaze_entry.get('frame_id')
            sa_entry = sa_by_frame.get(frame_id, {})
            
            # Combina dados
            combined_entry = {
                'participant_id': self.participant_id or 'unknown',
                'scenario_type': self.scenario_type or 'default',
                'frame_id': frame_id,
                'timestamp': gaze_entry.get('timestamp', 0),
                'sa_score': sa_entry.get('overall_score', ''),
                'sa_classification': sa_entry.get('classification', ''),
                'attention_zone': sa_entry.get('attention_zone', ''),
                'fatigue_level': sa_entry.get('fatigue_level', ''),
                'fixation_duration': sa_entry.get('fixation_duration', ''),
                'saccade_frequency': sa_entry.get('saccade_frequency', ''),
                'face_confidence': gaze_entry.get('face_confidence', ''),
                'processing_time': gaze_entry.get('processing_time', '')
            }
            
            combined_data.append(combined_entry)
        
        # Salva em CSV
        if combined_data:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=combined_data[0].keys())
                writer.writeheader()
                writer.writerows(combined_data)
            
            print(f"📊 Dados SPSS salvos: {filename}")
    
    def export_r_format(self):
        """Exporta dados em formato compatível com R"""
        # Similar ao SPSS, mas com algumas adaptações para R
        filename = f"{self.session_id}_r_format.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Usa o mesmo formato do SPSS (R pode ler CSV facilmente)
        self.export_spss_format()
        
        # Cria também script R básico
        r_script = f"""
# Script R para análise dos dados de eye-tracking
# Sessão: {self.session_id}

# Carregar dados
data <- read.csv("{filename}")

# Estatísticas descritivas básicas
summary(data)

# Análise de consciência situacional
sa_summary <- summary(data$sa_score)
print("Estatísticas de SA Score:")
print(sa_summary)

# Análise por zona de atenção
zone_table <- table(data$attention_zone)
print("Distribuição por zonas:")
print(zone_table)

# Análise de fadiga
fatigue_table <- table(data$fatigue_level)
print("Níveis de fadiga:")
print(fatigue_table)
"""
        
        script_filename = f"{self.session_id}_analysis.R"
        script_filepath = os.path.join(self.output_dir, script_filename)
        
        with open(script_filepath, 'w', encoding='utf-8') as f:
            f.write(r_script)
        
        print(f"📜 Script R salvo: {script_filename}")
    
    def export_matlab_format(self):
        """Exporta dados em formato compatível com MATLAB"""
        try:
            import scipy.io
            
            # Prepara dados para MATLAB
            matlab_data = {
                'session_id': self.session_id,
                'participant_id': self.participant_id or 'unknown',
                'scenario_type': self.scenario_type or 'default',
                'gaze_data': [],
                'sa_data': [],
                'events': []
            }
            
            # Converte dados de gaze
            for entry in self.gaze_buffer:
                gaze_vectors = entry.get('gaze_vectors', {})
                left_gaze = gaze_vectors.get('left', {})
                right_gaze = gaze_vectors.get('right', {})
                
                matlab_entry = [
                    entry.get('timestamp', 0),
                    left_gaze.get('yaw', 0),
                    left_gaze.get('pitch', 0),
                    right_gaze.get('yaw', 0),
                    right_gaze.get('pitch', 0),
                    entry.get('face_confidence', 0)
                ]
                matlab_data['gaze_data'].append(matlab_entry)
            
            # Converte dados de SA
            for entry in self.sa_buffer:
                sa_entry = [
                    entry.get('timestamp', 0),
                    entry.get('overall_score', 0),
                    entry.get('zone_confidence', 0),
                    entry.get('fatigue_score', 0),
                    entry.get('fixation_duration', 0),
                    entry.get('saccade_frequency', 0)
                ]
                matlab_data['sa_data'].append(sa_entry)
            
            # Salva arquivo .mat
            filename = f"{self.session_id}_matlab_data.mat"
            filepath = os.path.join(self.output_dir, filename)
            
            scipy.io.savemat(filepath, matlab_data)
            print(f"🔬 Dados MATLAB salvos: {filename}")
            
        except ImportError:
            print("⚠️ SciPy não disponível - instalando dados em CSV para MATLAB")
            self.export_spss_format()  # Fallback para CSV
        except Exception as e:
            print(f"❌ Erro na exportação MATLAB: {e}")
    
    def generate_summary_report(self) -> str:
        """Gera relatório resumo da sessão"""
        stats = self.session_data.get('session_statistics', {})
        session_info = self.session_data['session_info']
        
        report = f"""
=== RELATÓRIO DE SESSÃO DE EYE-TRACKING ===

Informações da Sessão:
- ID da Sessão: {self.session_id}
- Participante: {self.participant_id or 'N/A'}
- Cenário: {self.scenario_type or 'N/A'}
- Duração: {stats.get('duration_seconds', 0):.1f} segundos
- Frames processados: {stats.get('total_frames', 0)}

Análise de Consciência Situacional:
- Análises realizadas: {stats.get('total_sa_analyses', 0)}
- Score médio: {stats.get('sa_statistics', {}).get('average_sa_score', 0):.3f}
- Score mínimo: {stats.get('sa_statistics', {}).get('min_sa_score', 0):.3f}
- Score máximo: {stats.get('sa_statistics', {}).get('max_sa_score', 0):.3f}

Distribuição de Atenção:
"""
        
        # Adiciona distribuição de zonas
        zone_dist = stats.get('attention_zone_distribution', {})
        for zone, count in zone_dist.items():
            percentage = (count / stats.get('total_sa_analyses', 1)) * 100
            report += f"- {zone}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
Eventos Detectados:
- Total de eventos: {stats.get('total_events', 0)}
"""
        
        # Adiciona distribuição de eventos
        event_dist = stats.get('event_distribution', {})
        for event_type, count in event_dist.items():
            report += f"- {event_type}: {count}\n"
        
        report += f"""
Configuração do Sistema:
- Método de detecção facial: {session_info.get('system_config', {}).get('algorithm', {}).get('face_detection_method', 'N/A')}
- Método de detecção de íris: {session_info.get('system_config', {}).get('algorithm', {}).get('iris_detection_method', 'N/A')}
- Método de calibração: {session_info.get('system_config', {}).get('calibration', {}).get('method', 'N/A')}

=====================================
"""
        
        return report
    
    def save_summary_report(self):
        """Salva relatório resumo em arquivo"""
        report = self.generate_summary_report()
        
        filename = f"{self.session_id}_summary_report.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📋 Relatório resumo salvo: {filename}")
        return report

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import time


class GazeDataLogger:
    """Logger que salva dados de gaze tracking em Excel"""

    def __init__(self, output_dir="output/logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Nome do arquivo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.excel_file = self.output_dir / f"gaze_log_{timestamp}.xlsx"

        # Lista para armazenar dados
        self.data_buffer = []

        # Contador de frames
        self.frame_count = 0

        # Timestamp inicial
        self.start_time = time.time()

        # Configuração de autosave
        self.autosave_interval = 100  # Salvar a cada 100 frames

    def log_frame(self, result_data):
        """
        Registra dados de um frame processado
        """
        self.frame_count += 1
        current_time = time.time() - self.start_time

        # Extrair dados importantes
        row_data = {
            'frame': self.frame_count,
            'timestamp': current_time,
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],

            # Dados de posição na tela
            'screen_x': None,
            'screen_y': None,
            'filtered_x': None,
            'filtered_y': None,

            # Dados de gaze
            'left_yaw': None,
            'left_pitch': None,
            'right_yaw': None,
            'right_pitch': None,
            'avg_yaw': None,
            'avg_pitch': None,

            # Dados de qualidade
            'face_confidence': result_data.get('face_confidence', 0.0),
            'spheres_calibrated': result_data.get('spheres_calibrated', False),
            'discrepancy_angle': None,
            'gaze_quality': None,
            'use_single_eye': None,

            # Dados de performance
            'processing_time_ms': result_data.get('processing_time', 0.0),
            'scale_ratio': result_data.get('scale_ratio', None),

            # Dados 3D
            'head_center_x': None,
            'head_center_y': None,
            'head_center_z': None,
        }

        # Screen point
        if result_data.get('screen_point'):
            row_data['screen_x'] = result_data['screen_point'][0]
            row_data['screen_y'] = result_data['screen_point'][1]

        # Filtered point
        if result_data.get('filtered_point'):
            row_data['filtered_x'] = result_data['filtered_point'][0]
            row_data['filtered_y'] = result_data['filtered_point'][1]

        # Gaze vectors
        gaze_vectors = result_data.get('gaze_vectors', {})
        if 'left' in gaze_vectors:
            row_data['left_yaw'] = gaze_vectors['left'].get('yaw', None)
            row_data['left_pitch'] = gaze_vectors['left'].get('pitch', None)

        if 'right' in gaze_vectors:
            row_data['right_yaw'] = gaze_vectors['right'].get('yaw', None)
            row_data['right_pitch'] = gaze_vectors['right'].get('pitch', None)

        # Média
        if 'left' in gaze_vectors and 'right' in gaze_vectors:
            row_data['avg_yaw'] = (gaze_vectors['left'].get('yaw', 0) +
                                   gaze_vectors['right'].get('yaw', 0)) / 2
            row_data['avg_pitch'] = (gaze_vectors['left'].get('pitch', 0) +
                                     gaze_vectors['right'].get('pitch', 0)) / 2

        # Dados 3D de qualidade
        gaze_3d = result_data.get('gaze_vectors_3d', {})
        if gaze_3d:
            row_data['discrepancy_angle'] = gaze_3d.get('discrepancy_angle', None)
            row_data['gaze_quality'] = gaze_3d.get('quality', None)
            row_data['use_single_eye'] = gaze_3d.get('use_single_eye', None)

        # Head center
        if result_data.get('head_center') is not None:
            head_center = result_data['head_center']
            row_data['head_center_x'] = head_center[0]
            row_data['head_center_y'] = head_center[1]
            row_data['head_center_z'] = head_center[2]

        # Adicionar ao buffer
        self.data_buffer.append(row_data)

        # Autosave
        if self.frame_count % self.autosave_interval == 0:
            self.save()

    def save(self):
        """Salva os dados acumulados em Excel"""
        if not self.data_buffer:
            return

        try:
            # Criar DataFrame
            df = pd.DataFrame(self.data_buffer)

            # Salvar em Excel
            with pd.ExcelWriter(self.excel_file, engine='openpyxl') as writer:
                # Aba principal com todos os dados
                df.to_excel(writer, sheet_name='Medições', index=False)

                # Aba de estatísticas
                stats_df = self._calculate_statistics(df)
                stats_df.to_excel(writer, sheet_name='Estatísticas', index=False)

                # Aba de resumo de qualidade
                quality_df = self._calculate_quality_summary(df)
                quality_df.to_excel(writer, sheet_name='Qualidade', index=False)

        except Exception as e:
            # Fallback: salvar como CSV se Excel falhar
            csv_file = self.excel_file.with_suffix('.csv')
            df = pd.DataFrame(self.data_buffer)
            df.to_csv(csv_file, index=False)

    def _calculate_statistics(self, df):
        """Calcula estatísticas descritivas dos dados"""
        stats = []

        # Colunas numéricas para análise
        numeric_cols = ['screen_x', 'screen_y', 'filtered_x', 'filtered_y',
                       'left_yaw', 'left_pitch', 'right_yaw', 'right_pitch',
                       'avg_yaw', 'avg_pitch', 'face_confidence',
                       'processing_time_ms', 'discrepancy_angle']

        for col in numeric_cols:
            if col in df.columns and df[col].notna().any():
                stats.append({
                    'Métrica': col,
                    'Média': df[col].mean(),
                    'Mediana': df[col].median(),
                    'Desvio Padrão': df[col].std(),
                    'Mínimo': df[col].min(),
                    'Máximo': df[col].max(),
                    'Valores Válidos': df[col].notna().sum(),
                    'Valores Nulos': df[col].isna().sum()
                })

        return pd.DataFrame(stats)

    def _calculate_quality_summary(self, df):
        """Calcula resumo de qualidade do gaze tracking"""
        summary = []

        # Taxa de detecção
        total_frames = len(df)
        valid_screen_points = df['screen_x'].notna().sum()

        summary.append({
            'Métrica': 'Taxa de Detecção',
            'Valor': f"{(valid_screen_points/total_frames*100):.2f}%",
            'Detalhes': f"{valid_screen_points}/{total_frames} frames"
        })

        # Qualidade do gaze
        if 'gaze_quality' in df.columns:
            quality_counts = df['gaze_quality'].value_counts()
            for quality, count in quality_counts.items():
                if quality:
                    summary.append({
                        'Métrica': f'Qualidade: {quality}',
                        'Valor': f"{(count/total_frames*100):.2f}%",
                        'Detalhes': f"{count}/{total_frames} frames"
                    })

        # Taxa de calibração
        if 'spheres_calibrated' in df.columns:
            calibrated = df['spheres_calibrated'].sum()
            summary.append({
                'Métrica': 'Taxa de Calibração',
                'Valor': f"{(calibrated/total_frames*100):.2f}%",
                'Detalhes': f"{calibrated}/{total_frames} frames"
            })

        # FPS médio
        if 'processing_time_ms' in df.columns:
            avg_processing = df['processing_time_ms'].mean()
            avg_fps = 1000 / avg_processing if avg_processing > 0 else 0
            summary.append({
                'Métrica': 'FPS Médio de Processamento',
                'Valor': f"{avg_fps:.2f}",
                'Detalhes': f"Tempo médio: {avg_processing:.2f}ms"
            })

        # Discrepância média
        if 'discrepancy_angle' in df.columns and df['discrepancy_angle'].notna().any():
            avg_discrepancy = df['discrepancy_angle'].mean()
            summary.append({
                'Métrica': 'Discrepância Média Entre Olhos',
                'Valor': f"{avg_discrepancy:.2f}°",
                'Detalhes': f"Min: {df['discrepancy_angle'].min():.2f}°, Max: {df['discrepancy_angle'].max():.2f}°"
            })

        return pd.DataFrame(summary)

    def get_summary(self):
        """Retorna um resumo textual da sessão"""
        if not self.data_buffer:
            return "Nenhum dado registrado ainda."

        df = pd.DataFrame(self.data_buffer)

        valid_points = df['screen_x'].notna().sum()
        total_frames = len(df)
        detection_rate = (valid_points / total_frames * 100) if total_frames > 0 else 0

        duration = time.time() - self.start_time

        summary = f"""
╔════════════════════════════════════════════╗
║     RESUMO DA SESSÃO DE GAZE TRACKING      ║
╚════════════════════════════════════════════╝

📊 Dados Gerais:
   • Frames processados: {total_frames}
   • Duração: {duration:.2f}s
   • FPS médio: {total_frames/duration:.2f}

📍 Detecção:
   • Taxa de detecção: {detection_rate:.2f}%
   • Pontos válidos: {valid_points}/{total_frames}

💾 Arquivo:
   • {self.excel_file}
"""
        return summary

    def close(self):
        """Fecha o logger e salva os dados finais"""
        self.save()
        return self.get_summary()


class SessionLogger:
    """Logger simplificado para dados de sessão"""

    def __init__(self, output_dir="output/sessions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.output_dir / f"session_{timestamp}.txt"

        self.events = []
        self.start_time = time.time()

    def log_event(self, event_type, description):
        """Registra um evento da sessão"""
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.events.append({
            'time': elapsed,
            'timestamp': timestamp,
            'type': event_type,
            'description': description
        })

    def save(self):
        """Salva eventos da sessão"""
        with open(self.session_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("SESSÃO DE GAZE TRACKING\n")
            f.write(f"Iniciada em: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            for event in self.events:
                f.write(f"[{event['timestamp']}] ({event['time']:.2f}s) ")
                f.write(f"{event['type']}: {event['description']}\n")

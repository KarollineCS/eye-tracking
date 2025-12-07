import cv2
import numpy as np
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from datetime import datetime
import os


@dataclass
class GazeSample:
    """Uma amostra de gaze durante a validação"""
    timestamp: float          # Tempo desde início da captura
    screen_x: int            # Coordenada X na tela
    screen_y: int            # Coordenada Y na tela
    normalized_a: float      # Coordenada normalizada horizontal [0-1]
    normalized_b: float      # Coordenada normalizada vertical [0-1]
    cell_row: int            # Linha da célula onde caiu
    cell_col: int            # Coluna da célula onde caiu
    target_row: int          # Linha da célula alvo
    target_col: int          # Coluna da célula alvo
    distance_to_target: float  # Distância até o centro do alvo (pixels)


@dataclass
class CellValidationResult:
    """Resultado da validação de uma célula"""
    row: int
    col: int
    samples: List[GazeSample] = field(default_factory=list)
    
    @property
    def num_samples(self) -> int:
        return len(self.samples)
    
    @property
    def mean_error_pixels(self) -> float:
        """Erro médio em pixels até o centro do alvo"""
        if not self.samples:
            return 0.0
        return np.mean([s.distance_to_target for s in self.samples])
    
    @property
    def std_error_pixels(self) -> float:
        """Desvio padrão do erro em pixels"""
        if len(self.samples) < 2:
            return 0.0
        return np.std([s.distance_to_target for s in self.samples])
    
    @property
    def accuracy_percentage(self) -> float:
        """Porcentagem de amostras que caíram na célula correta"""
        if not self.samples:
            return 0.0
        correct = sum(1 for s in self.samples 
                     if s.cell_row == s.target_row and s.cell_col == s.target_col)
        return (correct / len(self.samples)) * 100
    
    @property
    def center_of_gaze(self) -> Tuple[float, float]:
        """Centro médio do gaze durante a captura"""
        if not self.samples:
            return (0.5, 0.5)
        mean_a = np.mean([s.normalized_a for s in self.samples])
        mean_b = np.mean([s.normalized_b for s in self.samples])
        return (mean_a, mean_b)


@dataclass
class ValidationSession:
    """Sessão completa de validação"""
    start_time: datetime
    screen_width: int
    screen_height: int
    grid_rows: int
    grid_cols: int
    capture_duration: float
    cell_results: Dict[Tuple[int, int], CellValidationResult] = field(default_factory=dict)
    
    def add_cell_result(self, result: CellValidationResult):
        self.cell_results[(result.row, result.col)] = result
    
    @property
    def overall_accuracy(self) -> float:
        """Precisão geral (% de amostras na célula correta)"""
        total_samples = 0
        correct_samples = 0
        for result in self.cell_results.values():
            for s in result.samples:
                total_samples += 1
                if s.cell_row == s.target_row and s.cell_col == s.target_col:
                    correct_samples += 1
        return (correct_samples / total_samples * 100) if total_samples > 0 else 0.0
    
    @property
    def overall_mean_error(self) -> float:
        """Erro médio geral em pixels"""
        all_errors = []
        for result in self.cell_results.values():
            all_errors.extend([s.distance_to_target for s in result.samples])
        return np.mean(all_errors) if all_errors else 0.0
    
    @property
    def overall_std_error(self) -> float:
        """Desvio padrão geral em pixels"""
        all_errors = []
        for result in self.cell_results.values():
            all_errors.extend([s.distance_to_target for s in result.samples])
        return np.std(all_errors) if len(all_errors) > 1 else 0.0


class ChessboardValidator:
    """
    Sistema de validação com chessboard para medir precisão do eye tracking.
    """
    
    def __init__(self, 
                 screen_width: int,
                 screen_height: int,
                 grid_rows: int = 3,
                 grid_cols: int = 4,
                 capture_duration: float = 10.0,
                 countdown_seconds: int = 3):
        """
        Args:
            screen_width: Largura da tela em pixels
            screen_height: Altura da tela em pixels
            grid_rows: Número de linhas do grid (padrão: 3)
            grid_cols: Número de colunas do grid (padrão: 4)
            capture_duration: Duração da captura por célula em segundos
            countdown_seconds: Segundos de contagem regressiva
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.capture_duration = capture_duration
        self.countdown_seconds = countdown_seconds
        
        # Calcular dimensões das células
        self.cell_width = screen_width // grid_cols
        self.cell_height = screen_height // grid_rows
        
        # Estado da validação
        self.is_running = False
        self.current_session: Optional[ValidationSession] = None
        self.current_target: Optional[Tuple[int, int]] = None
        self.capture_start_time: Optional[float] = None
        self.current_samples: List[GazeSample] = []
        
        # Ordem das células a validar (será embaralhada)
        self.cells_to_validate: List[Tuple[int, int]] = []
        self.cells_validated: List[Tuple[int, int]] = []
        
        # Estado da UI
        self.state = "idle"  # idle, waiting_ready, countdown, capturing, finished
        self.countdown_value = 0
        self.countdown_start = 0
        
        # Cores
        self.COLOR_LIGHT = (240, 240, 240)
        self.COLOR_DARK = (180, 180, 180)
        self.COLOR_TARGET = (0, 200, 0)
        self.COLOR_TARGET_BORDER = (0, 255, 0)
        self.COLOR_GAZE = (0, 0, 255)
        self.COLOR_TEXT = (50, 50, 50)
        self.COLOR_COUNTDOWN = (0, 100, 255)
        
    def get_cell_bounds(self, row: int, col: int) -> Tuple[int, int, int, int]:
        """Retorna (x1, y1, x2, y2) da célula"""
        x1 = col * self.cell_width
        y1 = row * self.cell_height
        x2 = x1 + self.cell_width
        y2 = y1 + self.cell_height
        return (x1, y1, x2, y2)
    
    def get_cell_center(self, row: int, col: int) -> Tuple[int, int]:
        """Retorna o centro da célula em pixels"""
        x1, y1, x2, y2 = self.get_cell_bounds(row, col)
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_cell_from_point(self, x: int, y: int) -> Tuple[int, int]:
        """Retorna (row, col) da célula que contém o ponto"""
        col = min(max(0, x // self.cell_width), self.grid_cols - 1)
        row = min(max(0, y // self.cell_height), self.grid_rows - 1)
        return (row, col)
    
    def start_session(self, validate_all: bool = True, 
                     specific_cells: List[Tuple[int, int]] = None):
        """
        Inicia uma nova sessão de validação.
        
        Args:
            validate_all: Se True, valida todas as células
            specific_cells: Lista de células específicas [(row, col), ...]
        """
        self.current_session = ValidationSession(
            start_time=datetime.now(),
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            capture_duration=self.capture_duration
        )
        
        # Definir células a validar
        if specific_cells:
            self.cells_to_validate = list(specific_cells)
        elif validate_all:
            self.cells_to_validate = [
                (r, c) for r in range(self.grid_rows) 
                for c in range(self.grid_cols)
            ]
        
        # Embaralhar ordem
        random.shuffle(self.cells_to_validate)
        self.cells_validated = []
        
        self.is_running = True
        self.state = "waiting_ready"
        self._select_next_target()
    
    def _select_next_target(self):
        """Seleciona a próxima célula alvo"""
        if self.cells_to_validate:
            self.current_target = self.cells_to_validate.pop(0)
            self.current_samples = []
            self.state = "waiting_ready"
        else:
            self.state = "finished"
            self.current_target = None
    
    def user_ready(self):
        """Chamado quando o usuário pressiona ESPAÇO para iniciar"""
        if self.state == "waiting_ready":
            self.state = "countdown"
            self.countdown_value = self.countdown_seconds
            self.countdown_start = time.time()
    
    def add_gaze_sample(self, screen_x: int, screen_y: int, 
                        normalized_a: float, normalized_b: float):
        """
        Adiciona uma amostra de gaze durante a captura.
        
        Args:
            screen_x, screen_y: Coordenadas na tela
            normalized_a, normalized_b: Coordenadas normalizadas [0-1]
        """
        if self.state != "capturing" or self.current_target is None:
            return
        
        target_row, target_col = self.current_target
        cell_row, cell_col = self.get_cell_from_point(screen_x, screen_y)
        
        # Calcular distância até o centro do alvo
        target_center = self.get_cell_center(target_row, target_col)
        distance = np.sqrt((screen_x - target_center[0])**2 + 
                          (screen_y - target_center[1])**2)
        
        sample = GazeSample(
            timestamp=time.time() - self.capture_start_time,
            screen_x=screen_x,
            screen_y=screen_y,
            normalized_a=normalized_a,
            normalized_b=normalized_b,
            cell_row=cell_row,
            cell_col=cell_col,
            target_row=target_row,
            target_col=target_col,
            distance_to_target=distance
        )
        
        self.current_samples.append(sample)
    
    def update(self) -> str:
        """
        Atualiza o estado da validação.
        Deve ser chamado a cada frame.
        
        Returns:
            Estado atual: "waiting_ready", "countdown", "capturing", "finished"
        """
        if self.state == "countdown":
            elapsed = time.time() - self.countdown_start
            remaining = self.countdown_seconds - int(elapsed)
            
            if remaining <= 0:
                # Iniciar captura
                self.state = "capturing"
                self.capture_start_time = time.time()
                self.countdown_value = 0
            else:
                self.countdown_value = remaining
                
        elif self.state == "capturing":
            elapsed = time.time() - self.capture_start_time
            
            if elapsed >= self.capture_duration:
                # Finalizar captura desta célula
                self._finish_current_cell()
                self._select_next_target()
        
        return self.state
    
    def _finish_current_cell(self):
        """Finaliza a captura da célula atual e salva resultados"""
        if self.current_target is None:
            return
        
        result = CellValidationResult(
            row=self.current_target[0],
            col=self.current_target[1],
            samples=self.current_samples.copy()
        )
        
        self.current_session.add_cell_result(result)
        self.cells_validated.append(self.current_target)
    
    def skip_current_cell(self):
        """Pula a célula atual sem coletar dados"""
        if self.current_target:
            self.cells_validated.append(self.current_target)
        self._select_next_target()
    
    def cancel_session(self):
        """Cancela a sessão atual"""
        self.is_running = False
        self.state = "idle"
        self.current_target = None
    
    def render_chessboard(self, 
                          current_gaze: Optional[Tuple[int, int]] = None,
                          show_gaze: bool = True) -> np.ndarray:
        """
        Renderiza o chessboard com estado atual.
        
        Args:
            current_gaze: Posição atual do gaze (x, y) para desenhar
            show_gaze: Se True, mostra o ponto de gaze
            
        Returns:
            Imagem do chessboard
        """
        # Criar imagem
        img = np.ones((self.screen_height, self.screen_width, 3), dtype=np.uint8) * 255
        
        # Desenhar células do chessboard
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1, y1, x2, y2 = self.get_cell_bounds(row, col)
                
                # Cor alternada (padrão xadrez)
                is_light = (row + col) % 2 == 0
                color = self.COLOR_LIGHT if is_light else self.COLOR_DARK
                
                # Se é a célula alvo, usar cor verde
                if self.current_target == (row, col):
                    color = self.COLOR_TARGET
                
                # Preencher célula
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                
                # Borda
                border_color = (100, 100, 100)
                if self.current_target == (row, col):
                    border_color = self.COLOR_TARGET_BORDER
                    cv2.rectangle(img, (x1+2, y1+2), (x2-2, y2-2), border_color, 3)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1)
                
                # Número da célula (para referência)
                cell_num = row * self.grid_cols + col + 1
                center = self.get_cell_center(row, col)
                cv2.putText(img, str(cell_num), 
                           (center[0] - 15, center[1] + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
        
        # Desenhar ponto de gaze atual
        if show_gaze and current_gaze is not None:
            gx, gy = current_gaze
            if 0 <= gx < self.screen_width and 0 <= gy < self.screen_height:
                cv2.circle(img, (gx, gy), 15, self.COLOR_GAZE, -1)
                cv2.circle(img, (gx, gy), 15, (255, 255, 255), 2)
        
        # Desenhar informações de estado
        self._draw_status_overlay(img)
        
        return img
    
    def _draw_status_overlay(self, img: np.ndarray):
        """Desenha informações de estado na imagem"""
        h, w = img.shape[:2]
        
        # Área de status no topo
        overlay_height = 80
        cv2.rectangle(img, (0, 0), (w, overlay_height), (50, 50, 50), -1)
        
        # Progresso
        total_cells = len(self.cells_validated) + len(self.cells_to_validate)
        if self.current_target:
            total_cells += 1
        progress_text = f"Progresso: {len(self.cells_validated)}/{total_cells}"
        cv2.putText(img, progress_text, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Estado específico
        if self.state == "waiting_ready":
            status_text = "Olhe para a celula VERDE e pressione ESPACO"
            cv2.putText(img, status_text, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            
            # Instrução adicional
            cv2.putText(img, "[ESC] Cancelar  |  [S] Pular celula", 
                       (w - 400, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
        elif self.state == "countdown":
            # Countdown grande no centro
            center_x, center_y = w // 2, h // 2
            cv2.putText(img, str(self.countdown_value), 
                       (center_x - 50, center_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 5.0, self.COLOR_COUNTDOWN, 10)
            
            status_text = "Preparando..."
            cv2.putText(img, status_text, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
            
        elif self.state == "capturing":
            elapsed = time.time() - self.capture_start_time
            remaining = max(0, self.capture_duration - elapsed)
            
            # Barra de progresso
            bar_width = 300
            bar_x = w - bar_width - 20
            progress = elapsed / self.capture_duration
            cv2.rectangle(img, (bar_x, 20), (bar_x + bar_width, 50), (100, 100, 100), -1)
            cv2.rectangle(img, (bar_x, 20), (bar_x + int(bar_width * progress), 50), 
                         (0, 255, 0), -1)
            
            status_text = f"CAPTURANDO... {remaining:.1f}s restantes | {len(self.current_samples)} amostras"
            cv2.putText(img, status_text, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            
        elif self.state == "finished":
            status_text = "VALIDACAO CONCLUIDA! Pressione qualquer tecla para ver resultados"
            cv2.putText(img, status_text, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    
    def generate_heatmap(self, 
                         cell_result: CellValidationResult = None,
                         all_samples: bool = False,
                         resolution: Tuple[int, int] = None) -> np.ndarray:
        """
        Gera um heatmap das amostras de gaze.
        
        Args:
            cell_result: Resultado de uma célula específica (ou None para todas)
            all_samples: Se True, usa amostras de todas as células
            resolution: Resolução do heatmap (padrão: tamanho da tela)
            
        Returns:
            Imagem do heatmap
        """
        if resolution is None:
            resolution = (self.screen_width, self.screen_height)
        
        w, h = resolution
        
        # Coletar amostras
        samples = []
        if cell_result:
            samples = cell_result.samples
        elif all_samples and self.current_session:
            for result in self.current_session.cell_results.values():
                samples.extend(result.samples)
        
        if not samples:
            # Retornar imagem vazia
            return np.zeros((h, w, 3), dtype=np.uint8)
        
        # Criar matriz de densidade
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Kernel gaussiano para suavização
        kernel_size = 51
        sigma = 20
        
        for sample in samples:
            # Mapear para resolução do heatmap
            x = int(sample.normalized_a * w)
            y = int(sample.normalized_b * h)
            
            # Clamp
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            
            # Adicionar ponto
            heatmap[y, x] += 1.0
        
        # Aplicar blur gaussiano
        heatmap = cv2.GaussianBlur(heatmap, (kernel_size, kernel_size), sigma)
        
        # Normalizar
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Converter para colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        return heatmap_colored
    
    def generate_report_image(self) -> np.ndarray:
        """
        Gera uma imagem com o relatório completo da validação.
        
        Returns:
            Imagem com heatmap + métricas
        """
        if not self.current_session or not self.current_session.cell_results:
            return np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Dimensões
        report_width = 1200
        report_height = 900
        
        img = np.ones((report_height, report_width, 3), dtype=np.uint8) * 255
        
        # Título
        cv2.putText(img, "RELATORIO DE VALIDACAO - EYE TRACKING", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        timestamp = self.current_session.start_time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, f"Data: {timestamp}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # Heatmap geral (lado esquerdo)
        heatmap = self.generate_heatmap(all_samples=True, resolution=(500, 375))
        
        # Desenhar grid no heatmap
        hm_h, hm_w = heatmap.shape[:2]
        cell_w = hm_w // self.grid_cols
        cell_h = hm_h // self.grid_rows
        
        for i in range(1, self.grid_cols):
            cv2.line(heatmap, (i * cell_w, 0), (i * cell_w, hm_h), (255, 255, 255), 1)
        for i in range(1, self.grid_rows):
            cv2.line(heatmap, (0, i * cell_h), (hm_w, i * cell_h), (255, 255, 255), 1)
        
        # Marcar centros dos alvos
        for (row, col), result in self.current_session.cell_results.items():
            cx = col * cell_w + cell_w // 2
            cy = row * cell_h + cell_h // 2
            cv2.drawMarker(heatmap, (cx, cy), (255, 255, 255), 
                          cv2.MARKER_CROSS, 20, 2)
        
        # Colocar heatmap na imagem
        img[100:100+375, 20:20+500] = heatmap
        cv2.rectangle(img, (18, 98), (522, 477), (0, 0, 0), 2)
        cv2.putText(img, "Heatmap Geral", (200, 500),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Métricas gerais (lado direito)
        metrics_x = 560
        y = 120
        
        cv2.putText(img, "METRICAS GERAIS", (metrics_x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y += 40
        
        session = self.current_session
        
        metrics = [
            f"Celulas validadas: {len(session.cell_results)}/{self.grid_rows * self.grid_cols}",
            f"Total de amostras: {sum(r.num_samples for r in session.cell_results.values())}",
            f"",
            f"Precisao geral: {session.overall_accuracy:.1f}%",
            f"Erro medio: {session.overall_mean_error:.1f} pixels",
            f"Desvio padrao: {session.overall_std_error:.1f} pixels",
        ]
        
        for metric in metrics:
            cv2.putText(img, metric, (metrics_x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
            y += 30
        
        # Tabela de resultados por célula
        y = 520
        cv2.putText(img, "RESULTADOS POR CELULA", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y += 30
        
        # Cabeçalho da tabela
        headers = ["Celula", "Amostras", "Precisao", "Erro Medio", "Desvio"]
        col_widths = [80, 100, 100, 120, 100]
        x = 20
        for header, width in zip(headers, col_widths):
            cv2.putText(img, header, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            x += width
        y += 25
        cv2.line(img, (20, y - 10), (520, y - 10), (0, 0, 0), 1)
        
        # Dados das células
        for (row, col), result in sorted(self.current_session.cell_results.items()):
            x = 20
            cell_name = f"({row+1},{col+1})"
            values = [
                cell_name,
                str(result.num_samples),
                f"{result.accuracy_percentage:.1f}%",
                f"{result.mean_error_pixels:.1f}px",
                f"{result.std_error_pixels:.1f}px"
            ]
            
            # Cor baseada na precisão
            accuracy = result.accuracy_percentage
            if accuracy >= 80:
                color = (0, 150, 0)
            elif accuracy >= 50:
                color = (0, 150, 150)
            else:
                color = (0, 0, 200)
            
            for value, width in zip(values, col_widths):
                cv2.putText(img, value, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                x += width
            y += 25
            
            if y > report_height - 50:
                break
        
        # Legenda de cores
        y = report_height - 40
        cv2.putText(img, "Legenda: ", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "Verde >= 80%", (100, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
        cv2.putText(img, "Amarelo >= 50%", (220, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 150), 1)
        cv2.putText(img, "Vermelho < 50%", (360, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        
        return img
    
    def save_results(self, output_dir: str = "validation_results"):
        """
        Salva os resultados da validação em arquivos.
        
        Args:
            output_dir: Diretório de saída
        """
        if not self.current_session:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = self.current_session.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Salvar heatmap
        heatmap = self.generate_heatmap(all_samples=True)
        cv2.imwrite(f"{output_dir}/heatmap_{timestamp}.png", heatmap)
        
        # Salvar relatório visual
        report = self.generate_report_image()
        cv2.imwrite(f"{output_dir}/report_{timestamp}.png", report)
        
        # Salvar dados em CSV
        csv_path = f"{output_dir}/data_{timestamp}.csv"
        with open(csv_path, 'w') as f:
            f.write("timestamp,screen_x,screen_y,norm_a,norm_b,cell_row,cell_col,target_row,target_col,distance\n")
            for result in self.current_session.cell_results.values():
                for s in result.samples:
                    f.write(f"{s.timestamp:.3f},{s.screen_x},{s.screen_y},"
                           f"{s.normalized_a:.4f},{s.normalized_b:.4f},"
                           f"{s.cell_row},{s.cell_col},{s.target_row},{s.target_col},"
                           f"{s.distance_to_target:.2f}\n")
        
        # Salvar métricas resumidas
        metrics_path = f"{output_dir}/metrics_{timestamp}.txt"
        with open(metrics_path, 'w') as f:
            f.write("=== RELATÓRIO DE VALIDAÇÃO ===\n\n")
            f.write(f"Data: {self.current_session.start_time}\n")
            f.write(f"Resolução: {self.screen_width}x{self.screen_height}\n")
            f.write(f"Grid: {self.grid_rows}x{self.grid_cols}\n")
            f.write(f"Duração por célula: {self.capture_duration}s\n\n")
            
            f.write("=== MÉTRICAS GERAIS ===\n")
            f.write(f"Precisão geral: {self.current_session.overall_accuracy:.1f}%\n")
            f.write(f"Erro médio: {self.current_session.overall_mean_error:.1f} pixels\n")
            f.write(f"Desvio padrão: {self.current_session.overall_std_error:.1f} pixels\n\n")
            
            f.write("=== RESULTADOS POR CÉLULA ===\n")
            for (row, col), result in sorted(self.current_session.cell_results.items()):
                f.write(f"\nCélula ({row+1},{col+1}):\n")
                f.write(f"  Amostras: {result.num_samples}\n")
                f.write(f"  Precisão: {result.accuracy_percentage:.1f}%\n")
                f.write(f"  Erro médio: {result.mean_error_pixels:.1f}px\n")
                f.write(f"  Desvio padrão: {result.std_error_pixels:.1f}px\n")
        
        print(f"[Validação] Resultados salvos em: {output_dir}/")
        return output_dir

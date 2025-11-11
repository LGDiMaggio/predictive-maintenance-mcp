"""
Professional HTML Report Templates for Machinery Diagnostics

This module contains modern, responsive HTML templates for data visualization.
All templates are self-contained with inline CSS and use Plotly.js CDN for interactivity.
"""

from typing import Dict, List, Any, Optional


def get_base_template(
    title: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Base HTML template with professional styling.
    
    Args:
        title: Report title
        content: Main HTML content
        metadata: Optional metadata dict (stored as JSON in data attribute)
    
    Returns:
        Complete HTML document
    """
    import json
    
    metadata_json = json.dumps(metadata or {}, indent=2)
    metadata_section = f'<script type="application/json" id="report-metadata">\n{metadata_json}\n</script>'
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="Predictive Maintenance MCP Server">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --background: #f8f9fa;
            --card-background: #ffffff;
            --text-primary: #2c3e50;
            --text-secondary: #7f8c8d;
            --border-color: #e0e0e0;
            --shadow: 0 2px 8px rgba(0,0,0,0.1);
            --shadow-hover: 0 4px 16px rgba(0,0,0,0.15);
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--background);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 0;
            margin: 0;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 1rem;
            box-shadow: var(--shadow);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        
        .header-content {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header h1 {{
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        
        .header .subtitle {{
            opacity: 0.95;
            font-size: 1rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1rem;
        }}
        
        .card {{
            background: var(--card-background);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow);
            transition: box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            box-shadow: var(--shadow-hover);
        }}
        
        .card-title {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-primary);
            border-bottom: 2px solid var(--secondary-color);
            padding-bottom: 0.5rem;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}
        
        .info-item {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid var(--secondary-color);
        }}
        
        .info-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.25rem;
            font-weight: 600;
        }}
        
        .info-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
        }}
        
        .chart-container {{
            background: var(--card-background);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow);
        }}
        
        .footer {{
            text-align: center;
            padding: 2rem 1rem;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-color);
            margin-top: 3rem;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .badge-success {{
            background: var(--success-color);
            color: white;
        }}
        
        .badge-warning {{
            background: var(--warning-color);
            color: white;
        }}
        
        .badge-danger {{
            background: var(--danger-color);
            color: white;
        }}
        
        .badge-info {{
            background: var(--secondary-color);
            color: white;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.5rem;
            }}
            .info-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        @media print {{
            .header {{
                position: static;
            }}
            .card {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    {metadata_section}
    {content}
    <div class="footer">
        <p>Generated by <strong>Predictive Maintenance MCP Server</strong></p>
        <p style="font-size: 0.875rem; margin-top: 0.5rem; color: var(--text-secondary);">
            Professional machinery diagnostics and vibration analysis
        </p>
    </div>
</body>
</html>"""


def create_fft_report(
    signal_file: str,
    sampling_rate: float,
    frequencies: List[float],
    magnitudes_db: List[float],
    peaks: List[Dict[str, float]],
    metadata: Dict[str, Any]
) -> str:
    """
    Create professional FFT spectrum report.
    
    Args:
        signal_file: Signal filename
        sampling_rate: Sampling rate in Hz
        frequencies: Frequency array
        magnitudes_db: Magnitude array in dB
        peaks: List of detected peaks with 'frequency' and 'magnitude_db'
        metadata: Additional metadata
    
    Returns:
        Complete HTML report
    """
    # Info cards
    info_cards = f"""
    <div class="info-grid">
        <div class="info-item">
            <div class="info-label">Sampling Rate</div>
            <div class="info-value">{sampling_rate:.0f} Hz</div>
        </div>
        <div class="info-item">
            <div class="info-label">Frequency Range</div>
            <div class="info-value">0 - {max(frequencies):.0f} Hz</div>
        </div>
        <div class="info-item">
            <div class="info-label">Signal Length</div>
            <div class="info-value">{metadata.get('num_samples', 'N/A'):,}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Duration</div>
            <div class="info-value">{metadata.get('duration', 0):.2f} s</div>
        </div>
    </div>
    """
    
    # Peaks table
    peaks_html = "<div class='card'><h3 class='card-title'>🎯 Detected Peaks</h3><table style='width:100%; border-collapse: collapse;'>"
    peaks_html += "<tr style='background: #f5f7fa; font-weight: 600;'><th style='padding: 0.75rem; text-align: left;'>Rank</th><th style='padding: 0.75rem; text-align: left;'>Frequency (Hz)</th><th style='padding: 0.75rem; text-align: left;'>Magnitude (dB)</th><th style='padding: 0.75rem; text-align: left;'>Note</th></tr>"
    
    for i, peak in enumerate(peaks[:10], 1):
        freq = peak['frequency']
        mag_db = peak['magnitude_db']
        note = peak.get('note', '')
        
        peaks_html += f"<tr style='border-bottom: 1px solid #e0e0e0;'>"
        peaks_html += f"<td style='padding: 0.75rem;'><strong>#{i}</strong></td>"
        peaks_html += f"<td style='padding: 0.75rem; font-family: monospace;'>{freq:.2f}</td>"
        peaks_html += f"<td style='padding: 0.75rem; font-family: monospace;'>{mag_db:.1f}</td>"
        peaks_html += f"<td style='padding: 0.75rem; color: #e74c3c;'>{note}</td>"
        peaks_html += "</tr>"
    
    peaks_html += "</table></div>"
    
    # Plotly chart
    chart_div = "<div class='chart-container'><div id='fft-chart'></div></div>"
    
    # Plotly script
    peak_freqs = [p['frequency'] for p in peaks[:10]]
    peak_mags = [p['magnitude_db'] for p in peaks[:10]]
    peak_labels = [f"{p['frequency']:.1f}" for p in peaks[:10]]
    
    plotly_script = f"""
    <script>
        var spectrum = {{
            x: {frequencies},
            y: {magnitudes_db},
            type: 'scatter',
            mode: 'lines',
            name: 'Spectrum',
            line: {{
                color: '#667eea',
                width: 1.5
            }},
            hovertemplate: '%{{x:.2f}} Hz<br>%{{y:.1f}} dB<extra></extra>'
        }};
        
        var peaks = {{
            x: {peak_freqs},
            y: {peak_mags},
            type: 'scatter',
            mode: 'markers+text',
            name: 'Peaks',
            marker: {{
                color: '#e74c3c',
                size: 10,
                symbol: 'diamond',
                line: {{
                    color: 'white',
                    width: 2
                }}
            }},
            text: {peak_labels},
            textposition: 'top center',
            textfont: {{
                size: 10,
                color: '#e74c3c',
                family: 'monospace'
            }},
            hovertemplate: '%{{x:.2f}} Hz<br>%{{y:.1f}} dB<extra></extra>'
        }};
        
        var layout = {{
            title: {{
                text: 'FFT Spectrum Analysis',
                font: {{ size: 20, color: '#2c3e50' }}
            }},
            xaxis: {{
                title: 'Frequency (Hz)',
                gridcolor: '#e0e0e0',
                showgrid: true
            }},
            yaxis: {{
                title: 'Magnitude (dB re. max)',
                gridcolor: '#e0e0e0',
                showgrid: true
            }},
            hovermode: 'closest',
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            showlegend: true,
            legend: {{
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#ccc',
                borderwidth: 1
            }},
            margin: {{ t: 80, r: 30, b: 60, l: 70 }}
        }};
        
        var config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        }};
        
        Plotly.newPlot('fft-chart', [spectrum, peaks], layout, config);
    </script>
    """
    
    content = f"""
    <div class="header">
        <div class="header-content">
            <h1>📊 FFT Spectrum Analysis</h1>
            <p class="subtitle">{signal_file}</p>
        </div>
    </div>
    <div class="container">
        {info_cards}
        {chart_div}
        {peaks_html}
    </div>
    {plotly_script}
    """
    
    return get_base_template(
        title=f"FFT Analysis - {signal_file}",
        content=content,
        metadata=metadata
    )


def create_envelope_report(
    signal_file: str,
    sampling_rate: float,
    filter_band: tuple,
    time_data: List[float],
    filtered_signal: List[float],
    envelope: List[float],
    env_freq: List[float],
    env_mag_db: List[float],
    peaks: List[Dict[str, float]],
    bearing_freqs: Optional[Dict[str, float]],
    metadata: Dict[str, Any]
) -> str:
    """
    Create professional envelope analysis report.
    
    Args:
        signal_file: Signal filename
        sampling_rate: Sampling rate
        filter_band: (low, high) Hz
        time_data: Time array for signal plot
        filtered_signal: Filtered signal
        envelope: Envelope signal
        env_freq: Envelope spectrum frequencies
        env_mag_db: Envelope spectrum magnitudes in dB
        peaks: Detected peaks
        bearing_freqs: Optional dict with BPFO, BPFI, BSF, FTF
        metadata: Additional metadata
    
    Returns:
        Complete HTML report
    """
    # Info cards
    info_cards = f"""
    <div class="info-grid">
        <div class="info-item">
            <div class="info-label">Filter Range</div>
            <div class="info-value">{filter_band[0]:.0f}-{filter_band[1]:.0f} Hz</div>
        </div>
        <div class="info-item">
            <div class="info-label">Sampling Rate</div>
            <div class="info-value">{sampling_rate:.0f} Hz</div>
        </div>
        <div class="info-item">
            <div class="info-label">Signal Length</div>
            <div class="info-value">{metadata.get('num_samples', 'N/A'):,}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Duration</div>
            <div class="info-value">{metadata.get('duration', 0):.2f} s</div>
        </div>
    </div>
    """
    
    # Bearing frequencies reference (if provided)
    bearing_ref = ""
    if bearing_freqs:
        bearing_ref = "<div class='card'><h3 class='card-title'>📌 Bearing Characteristic Frequencies</h3><div class='info-grid'>"
        colors = {"BPFO": "#e74c3c", "BPFI": "#f39c12", "BSF": "#3498db", "FTF": "#2ecc71"}
        for name, freq in bearing_freqs.items():
            if freq:
                color = colors.get(name, "#95a5a6")
                bearing_ref += f"""
                <div class="info-item" style="border-left-color: {color};">
                    <div class="info-label">{name}</div>
                    <div class="info-value" style="color: {color};">{freq:.2f} Hz</div>
                </div>
                """
        bearing_ref += "</div></div>"
    
    # Peaks table
    peaks_html = "<div class='card'><h3 class='card-title'>🎯 Envelope Spectrum Peaks</h3><table style='width:100%; border-collapse: collapse;'>"
    peaks_html += "<tr style='background: #f5f7fa; font-weight: 600;'><th style='padding: 0.75rem; text-align: left;'>Rank</th><th style='padding: 0.75rem; text-align: left;'>Frequency (Hz)</th><th style='padding: 0.75rem; text-align: left;'>Magnitude (dB)</th><th style='padding: 0.75rem; text-align: left;'>Match</th></tr>"
    
    for i, peak in enumerate(peaks[:10], 1):
        freq = peak['frequency']
        mag_db = peak['magnitude_db']
        match = peak.get('match', '')
        
        peaks_html += f"<tr style='border-bottom: 1px solid #e0e0e0;'>"
        peaks_html += f"<td style='padding: 0.75rem;'><strong>#{i}</strong></td>"
        peaks_html += f"<td style='padding: 0.75rem; font-family: monospace;'>{freq:.2f}</td>"
        peaks_html += f"<td style='padding: 0.75rem; font-family: monospace;'>{mag_db:.1f}</td>"
        peaks_html += f"<td style='padding: 0.75rem; color: #e74c3c; font-weight: 600;'>{match}</td>"
        peaks_html += "</tr>"
    
    peaks_html += "</table></div>"
    
    # Charts
    charts_div = "<div class='chart-container'><div id='envelope-charts'></div></div>"
    
    # Plotly script with subplots
    peak_freqs = [p['frequency'] for p in peaks[:10]]
    peak_mags = [p['magnitude_db'] for p in peaks[:10]]
    
    # Bearing frequency markers
    bearing_markers_script = ""
    if bearing_freqs:
        for name, freq in bearing_freqs.items():
            if freq and freq <= max(env_freq):
                colors = {"BPFO": "#e74c3c", "BPFI": "#f39c12", "BSF": "#3498db", "FTF": "#2ecc71"}
                color = colors.get(name, "#95a5a6")
                bearing_markers_script += f"""
        data.push({{
            x: [{freq}, {freq}],
            y: [-60, 0],
            type: 'scatter',
            mode: 'lines',
            name: '{name}',
            line: {{ color: '{color}', width: 2, dash: 'dash' }},
            xaxis: 'x2',
            yaxis: 'y2',
            showlegend: true,
            hovertemplate: '{name}: {freq:.2f} Hz<extra></extra>'
        }});
        """
    
    plotly_script = f"""
    <script>
        var filtered = {{
            x: {time_data},
            y: {filtered_signal},
            type: 'scatter',
            mode: 'lines',
            name: 'Filtered Signal',
            line: {{ color: '#95a5a6', width: 0.8 }},
            xaxis: 'x',
            yaxis: 'y',
            hovertemplate: '%{{x:.3f}} s<br>%{{y:.4f}}<extra></extra>'
        }};
        
        var envelope_trace = {{
            x: {time_data},
            y: {envelope},
            type: 'scatter',
            mode: 'lines',
            name: 'Envelope',
            line: {{ color: '#e74c3c', width: 2 }},
            xaxis: 'x',
            yaxis: 'y',
            hovertemplate: '%{{x:.3f}} s<br>%{{y:.4f}}<extra></extra>'
        }};
        
        var spectrum = {{
            x: {env_freq},
            y: {env_mag_db},
            type: 'scatter',
            mode: 'lines',
            name: 'Envelope Spectrum',
            line: {{ color: '#11998e', width: 1.5 }},
            xaxis: 'x2',
            yaxis: 'y2',
            hovertemplate: '%{{x:.2f}} Hz<br>%{{y:.1f}} dB<extra></extra>'
        }};
        
        var peaks = {{
            x: {peak_freqs},
            y: {peak_mags},
            type: 'scatter',
            mode: 'markers',
            name: 'Peaks',
            marker: {{
                color: '#e74c3c',
                size: 10,
                symbol: 'circle',
                line: {{ color: 'white', width: 2 }}
            }},
            xaxis: 'x2',
            yaxis: 'y2',
            hovertemplate: '%{{x:.2f}} Hz<br>%{{y:.1f}} dB<extra></extra>'
        }};
        
        var data = [filtered, envelope_trace, spectrum, peaks];
        
        {bearing_markers_script}
        
        var layout = {{
            title: {{
                text: 'Envelope Analysis (Time + Frequency Domain)',
                font: {{ size: 20, color: '#2c3e50' }}
            }},
            grid: {{ rows: 2, columns: 1, subplots: [['xy'], ['x2y2']], roworder: 'top to bottom' }},
            xaxis: {{
                title: 'Time (s)',
                domain: [0, 1],
                anchor: 'y'
            }},
            yaxis: {{
                title: 'Amplitude',
                domain: [0.55, 1],
                anchor: 'x'
            }},
            xaxis2: {{
                title: 'Frequency (Hz)',
                domain: [0, 1],
                anchor: 'y2'
            }},
            yaxis2: {{
                title: 'Magnitude (dB re. max)',
                domain: [0, 0.45],
                anchor: 'x2',
                range: [-60, 5]
            }},
            hovermode: 'closest',
            showlegend: true,
            legend: {{
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#ccc',
                borderwidth: 1
            }},
            margin: {{ t: 80, r: 30, b: 60, l: 70 }},
            height: 800
        }};
        
        var config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        }};
        
        Plotly.newPlot('envelope-charts', data, layout, config);
    </script>
    """
    
    content = f"""
    <div class="header">
        <div class="header-content">
            <h1>📈 Envelope Analysis</h1>
            <p class="subtitle">{signal_file}</p>
        </div>
    </div>
    <div class="container">
        {info_cards}
        {bearing_ref}
        {charts_div}
        {peaks_html}
    </div>
    {plotly_script}
    """
    
    return get_base_template(
        title=f"Envelope Analysis - {signal_file}",
        content=content,
        metadata=metadata
    )


def create_iso_report(
    signal_file: str,
    iso_result: Dict[str, Any],
    metadata: Dict[str, Any]
) -> str:
    """
    Create professional ISO 20816-3 evaluation report.
    
    Args:
        signal_file: Signal filename
        iso_result: ISO evaluation result dict
        metadata: Additional metadata
    
    Returns:
        Complete HTML report
    """
    zone = iso_result['zone']
    severity = iso_result['severity_level']
    rms_velocity = iso_result['rms_velocity']
    
    # Zone color and icon
    zone_colors = {
        "A": ("#27ae60", "✓"),
        "B": ("#f39c12", "⚠"),
        "C": ("#e67e22", "⚠"),
        "D": ("#c0392b", "🚨")
    }
    color, icon = zone_colors.get(zone, ("#95a5a6", "?"))
    
    # Status badge
    status_badge = f"""
    <div style="text-align: center; margin: 2rem 0;">
        <div style="display: inline-block; padding: 1.5rem 3rem; border-radius: 20px; background: {color}; color: white; font-size: 1.5rem; font-weight: 700; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
            {icon} Zone {zone} - {severity}
        </div>
    </div>
    """
    
    # Info cards
    info_cards = f"""
    <div class="info-grid">
        <div class="info-item">
            <div class="info-label">RMS Velocity</div>
            <div class="info-value">{rms_velocity:.2f} mm/s</div>
        </div>
        <div class="info-item">
            <div class="info-label">Evaluation Zone</div>
            <div class="info-value" style="color: {color};">Zone {zone}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Machine Group</div>
            <div class="info-value">{iso_result['machine_group']}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Support Type</div>
            <div class="info-value">{iso_result['support_type'].title()}</div>
        </div>
    </div>
    """
    
    # Zone boundaries
    boundaries_card = f"""
    <div class="card">
        <h3 class="card-title">📏 Zone Boundaries</h3>
        <div class="info-grid">
            <div class="info-item" style="border-left-color: #27ae60;">
                <div class="info-label">Zone A/B</div>
                <div class="info-value">{iso_result['boundary_ab']:.1f} mm/s</div>
            </div>
            <div class="info-item" style="border-left-color: #f39c12;">
                <div class="info-label">Zone B/C</div>
                <div class="info-value">{iso_result['boundary_bc']:.1f} mm/s</div>
            </div>
            <div class="info-item" style="border-left-color: #e67e22;">
                <div class="info-label">Zone C/D</div>
                <div class="info-value">{iso_result['boundary_cd']:.1f} mm/s</div>
            </div>
        </div>
    </div>
    """
    
    # Interpretation
    interpretation_card = f"""
    <div class="card">
        <h3 class="card-title">💡 Interpretation</h3>
        <p style="font-size: 1.1rem; line-height: 1.8; color: var(--text-primary);">
            {iso_result['zone_description']}
        </p>
    </div>
    """
    
    # Chart
    chart_div = "<div class='chart-container'><div id='iso-chart'></div></div>"
    
    boundaries = [0, iso_result['boundary_ab'], iso_result['boundary_bc'], iso_result['boundary_cd'], iso_result['boundary_cd'] * 1.3]
    
    plotly_script = f"""
    <script>
        var boundaries = {{
            AB: {iso_result['boundary_ab']},
            BC: {iso_result['boundary_bc']},
            CD: {iso_result['boundary_cd']},
            max: {iso_result['boundary_cd'] * 1.3}
        }};
        
        var rmsVelocity = {rms_velocity};
        
        var trace1 = {{
            x: [boundaries.AB],
            y: ['Vibration Severity'],
            name: 'Zone A',
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: '#27ae60',
                line: {{ color: '#229954', width: 2 }}
            }},
            hovertemplate: 'Zone A: 0 - ' + boundaries.AB + ' mm/s<br>New machine condition<extra></extra>'
        }};
        
        var trace2 = {{
            x: [boundaries.BC - boundaries.AB],
            y: ['Vibration Severity'],
            name: 'Zone B',
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: '#f39c12',
                line: {{ color: '#e67e22', width: 2 }}
            }},
            hovertemplate: 'Zone B: ' + boundaries.AB + ' - ' + boundaries.BC + ' mm/s<br>Acceptable operation<extra></extra>'
        }};
        
        var trace3 = {{
            x: [boundaries.CD - boundaries.BC],
            y: ['Vibration Severity'],
            name: 'Zone C',
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: '#e67e22',
                line: {{ color: '#d35400', width: 2 }}
            }},
            hovertemplate: 'Zone C: ' + boundaries.BC + ' - ' + boundaries.CD + ' mm/s<br>Unsatisfactory<extra></extra>'
        }};
        
        var trace4 = {{
            x: [boundaries.max - boundaries.CD],
            y: ['Vibration Severity'],
            name: 'Zone D',
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: '#c0392b',
                line: {{ color: '#a93226', width: 2 }}
            }},
            hovertemplate: 'Zone D: > ' + boundaries.CD + ' mm/s<br>Severe condition<extra></extra>'
        }};
        
        var marker = {{
            x: [rmsVelocity],
            y: ['Vibration Severity'],
            mode: 'markers+text',
            type: 'scatter',
            name: 'Measured',
            marker: {{
                color: '#2c3e50',
                size: 20,
                symbol: 'circle',
                line: {{ color: '#fff', width: 3 }}
            }},
            text: [rmsVelocity.toFixed(2) + ' mm/s'],
            textposition: 'top center',
            textfont: {{ size: 14, color: '#2c3e50', family: 'Arial Black' }},
            hovertemplate: 'Measured RMS Velocity: ' + rmsVelocity.toFixed(2) + ' mm/s<extra></extra>'
        }};
        
        var data = [trace1, trace2, trace3, trace4, marker];
        
        var layout = {{
            title: {{
                text: 'Vibration Severity according to ISO 20816-3',
                font: {{ size: 20, color: '#2c3e50' }}
            }},
            barmode: 'stack',
            xaxis: {{
                title: 'RMS Velocity (mm/s)',
                range: [0, boundaries.max],
                showgrid: true,
                gridcolor: '#ecf0f1',
                zeroline: true
            }},
            yaxis: {{
                showticklabels: false
            }},
            height: 400,
            margin: {{ l: 50, r: 50, t: 80, b: 80 }},
            showlegend: true,
            legend: {{
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: -0.2
            }},
            hovermode: 'closest',
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white'
        }};
        
        var config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        }};
        
        Plotly.newPlot('iso-chart', data, layout, config);
    </script>
    """
    
    content = f"""
    <div class="header">
        <div class="header-content">
            <h1>📋 ISO 20816-3 Evaluation</h1>
            <p class="subtitle">{signal_file}</p>
        </div>
    </div>
    <div class="container">
        {status_badge}
        {info_cards}
        {chart_div}
        {boundaries_card}
        {interpretation_card}
    </div>
    {plotly_script}
    """
    
    return get_base_template(
        title=f"ISO 20816-3 - {signal_file}",
        content=content,
        metadata=metadata
    )

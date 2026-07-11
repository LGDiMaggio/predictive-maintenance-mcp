"""MCP tools for signal acquisition and data management (ISO 13374 Block 1)."""

import logging
import json
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP, Context

from ..config import DATA_DIR, MODELS_DIR, RESOURCES_DIR, REPORTS_DIR
from ..signal_loader import load_signal_data, SUPPORTED_EXTENSIONS, get_metadata_path
from ..signal_repository import get_repository
from ..models import StoredSignalInfo, SignalInfo
from ..document_reader import extract_text_from_pdf
from ._utils import safe_resolve

logger = logging.getLogger(__name__)


def register(mcp: FastMCP) -> None:
    """Register acquisition tools and resources on the given FastMCP instance."""

    # ------------------------------------------------------------------
    # RESOURCES
    # ------------------------------------------------------------------

    @mcp.resource("signal://list")
    def list_available_signals() -> str:
        """
        List all available signals in the data/signals directory.

        ⚠️ **CRITICAL - LLM Inference Policy:**
        - Filenames are OPAQUE IDENTIFIERS ONLY
        - NEVER infer signal content, fault type, or condition from filename
        - "baseline" in filename ≠ healthy signal
        - "InnerRaceFault" in filename ≠ inner race fault exists
        - "OuterRaceFault" in filename ≠ outer race fault exists
        - Filenames may be misleading or incorrect
        - Base ALL analysis on signal data evidence ONLY

        Returns:
            JSON with the list of available signal files
        """
        try:
            if not DATA_DIR.exists():
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                return "[]"

            signals = []
            for file_path in DATA_DIR.glob("**/*"):
                if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
                    signals.append({
                        "filename": str(file_path.relative_to(DATA_DIR)).replace("\\", "/"),
                        "path": str(file_path),
                        "size_bytes": file_path.stat().st_size,
                        "extension": file_path.suffix
                    })

            # Prepend critical warning to output
            warning = (
                "\n⚠️  CRITICAL: Filenames are OPAQUE IDENTIFIERS ONLY\n"
                "DO NOT infer fault type, condition, or signal characteristics from filenames.\n"
                "Filenames like 'baseline', 'InnerRaceFault', 'OuterRaceFault' may be MISLEADING.\n"
                "Base ALL diagnostics on signal data analysis ONLY.\n\n"
            )

            return warning + pd.DataFrame(signals).to_json(orient="records", indent=2)

        except Exception as e:
            logger.error(f"Error listing signals: {e}")
            return f'{{"error": "{str(e)}"}}'

    @mcp.resource("signal://read/{filename}")
    def read_signal_file(filename: str) -> str:
        """
        Read a signal file and return metadata summary.

        Supports formats: CSV, TXT, NPY, MAT (MATLAB), WAV (audio), Parquet

        NOTE: Returns metadata and statistics only — raw samples are NOT included
        to avoid context overflow. Use plot_signal() for visual inspection or
        load the signal in analysis tools (analyze_fft, analyze_envelope, etc.).

        Args:
            filename: Name of the file to read

        Returns:
            JSON with signal metadata and basic statistics (no raw data)
        """
        try:
            file_path = safe_resolve(DATA_DIR, filename)

            if not file_path.exists():
                return f'{{"error": "File {filename} not found"}}'

            # Use the unified loader for all supported formats
            signal_array = load_signal_data(filename)

            if signal_array is None:
                return f'{{"error": "Could not load signal from {filename}. Unsupported format or read error."}}'

            # Return metadata + statistics, NOT raw samples
            rms = float(np.sqrt(np.mean(signal_array**2)))
            result = {
                "filename": filename,
                "num_samples": len(signal_array),
                "duration_hint": f"{len(signal_array)} samples (provide sampling_rate for duration in seconds)",
                "statistics": {
                    "rms": round(rms, 6),
                    "peak": round(float(np.max(np.abs(signal_array))), 6),
                    "mean": round(float(np.mean(signal_array)), 6),
                    "std": round(float(np.std(signal_array)), 6),
                    "min": round(float(np.min(signal_array)), 6),
                    "max": round(float(np.max(signal_array)), 6),
                },
                "note": "Raw samples not included — use analysis tools or plot_signal() for inspection."
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error reading signal {filename}: {e}")
            return f'{{"error": "{str(e)}"}}'

    @mcp.resource("manual://list")
    def list_manuals_resource() -> str:
        """
        List all available machine manuals (PDF and TXT) as MCP resource.

        **IMPORTANT - LLM Usage Guidelines:**
        - This resource returns ONLY the list of available files
        - DO NOT make assumptions about manual content without reading it
        - DO NOT infer specifications from filenames alone
        - To access manual content, use manual://read/{filename} resource
        - If user asks about manual content, read it first before answering

        Returns:
            JSON with list of available manuals

        Example usage in Claude:
            "Show me what machine manuals are available"
        """
        manuals_dir = RESOURCES_DIR / "machine_manuals"
        manuals = []

        # Include both PDF and TXT files
        for manual_file in list(manuals_dir.glob("*.pdf")) + list(manuals_dir.glob("*.txt")):
            stat = manual_file.stat()
            manuals.append({
                "filename": manual_file.name,
                "type": manual_file.suffix.upper()[1:],  # PDF or TXT
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "uri": f"manual://read/{manual_file.name}"
            })

        result = {
            "total_manuals": len(manuals),
            "manuals": sorted(manuals, key=lambda x: x['filename']),
            "directory": str(manuals_dir.relative_to(RESOURCES_DIR.parent))
        }

        return json.dumps(result, indent=2)

    @mcp.resource("manual://read/{filename}")
    def read_manual_resource(filename: str) -> str:
        """
        Read machine manual (PDF or TXT) as text (MCP resource).

        Extracts text from first 20 pages (PDF) or full text (TXT) to avoid token overflow.
        For full manual, use read_manual_excerpt() tool with custom max_pages.

        **IMPORTANT - LLM Usage Guidelines:**
        - This resource returns ONLY the text from the manual file
        - Base ALL answers EXCLUSIVELY on the returned text content
        - DO NOT add information not present in the text
        - DO NOT make assumptions about missing information
        - If something is not in the text, say "Not found in the manual"
        - ALWAYS cite the manual when answering: "According to {filename}..."
        - If user needs complete manual, use read_manual_excerpt() tool

        Args:
            filename: Manual filename in resources/machine_manuals/ (PDF or TXT)

        Returns:
            Extracted text from manual

        Example usage in Claude:
            "Read the pump manual and tell me what bearings are specified"
        """
        try:
            manual_path = safe_resolve(RESOURCES_DIR / "machine_manuals", filename)

            if not manual_path.exists():
                return json.dumps({
                    "error": f"Manual not found: {filename}",
                    "available": [f.name for f in (RESOURCES_DIR / "machine_manuals").glob("*.pdf")] +
                                [f.name for f in (RESOURCES_DIR / "machine_manuals").glob("*.txt")]
                }, indent=2)

            # Read based on file type
            if manual_path.suffix.lower() == '.txt':
                with open(manual_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                pages_info = "full text file"
            else:
                # Extract text (limit to 20 pages to avoid token overflow)
                text = extract_text_from_pdf(manual_path, max_pages=20)
                pages_info = "first 20 pages"

            result = {
                "filename": filename,
                "content_type": manual_path.suffix.lower(),
                "pages_info": pages_info,
                "text_length": len(text),
                "text": text[:10000],  # First 10K chars
                "note": f"Showing {pages_info}. Use read_manual_excerpt() tool for full access."
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error reading manual {filename}: {e}")
            return json.dumps({"error": str(e)}, indent=2)

    # ------------------------------------------------------------------
    # TOOLS
    # ------------------------------------------------------------------

    @mcp.tool()
    def list_signals() -> str:
        """
        List all available signal files in the data/signals directory.

        Returns:
            A formatted text listing of all available signals
        """
        try:
            if not DATA_DIR.exists():
                return "No signals directory found"

            signals = []
            for file_path in DATA_DIR.glob("**/*"):
                if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
                    relative_path = file_path.relative_to(DATA_DIR)
                    signals.append(str(relative_path).replace("\\", "/"))

            if not signals:
                return "No signal files found"

            # Group by directory
            from collections import defaultdict
            by_dir = defaultdict(list)
            for sig in signals:
                dir_name = sig.split('/')[0] if '/' in sig else "root"
                by_dir[dir_name].append(sig)

            # Format output
            output = [f"Found {len(signals)} signal files:"]
            for dir_name in sorted(by_dir.keys()):
                files = by_dir[dir_name]
                output.append(f"\n{dir_name}/ ({len(files)} files):")
                for f in sorted(files):
                    output.append(f"  - {f}")

            return "\n".join(output)

        except Exception as e:
            return f"Error: {str(e)}"

    @mcp.tool()
    async def generate_test_signal(
        signal_type: str = "bearing_fault",
        duration: float = 10.0,
        sampling_rate: float = 10000.0,
        noise_level: float = 0.1,
        random_seed: Optional[int] = None,
        ctx: Context = None
    ) -> str:
        """
        Generate a test signal to validate analyses.

        Useful for testing algorithms without having real data available.

        Args:
            signal_type: Signal type ("bearing_fault", "gear_fault", "imbalance", "normal")
            duration: Signal duration in seconds (default: 10.0, gives 0.1 Hz frequency resolution)
            sampling_rate: Sampling frequency in Hz (default: 10000)
            noise_level: Noise level to add (default: 0.1)
            ctx: Context for logging

        Returns:
            Generated file name
        """
        if ctx:
            await ctx.info(f"Generating {signal_type} test signal...")

        rng = np.random.default_rng(random_seed)

        # Time parameters
        t = np.linspace(0, duration, int(sampling_rate * duration))

        # Generate signal based on type
        if signal_type == "bearing_fault":
            # Faulty bearing: periodic impulses + harmonics
            fault_freq = 10.0  # Hz - fault frequency
            carrier_freq = 1000.0  # Hz - carrier frequency

            # Periodic impulses
            impulses = np.zeros_like(t)
            impulse_times = np.arange(0, duration, 1/fault_freq)
            for imp_time in impulse_times:
                idx = np.argmin(np.abs(t - imp_time))
                impulses[idx] = 1.0

            # Convolution with impulse response
            impulse_response = np.exp(-50 * np.abs(t - t[len(t)//2]))
            signal_clean = np.convolve(impulses, impulse_response, mode='same')

            # Modulation with carrier
            signal_clean = signal_clean * np.sin(2 * np.pi * carrier_freq * t)

        elif signal_type == "gear_fault":
            # Faulty gear: component at mesh frequency
            mesh_freq = 200.0  # Hz
            signal_clean = np.sin(2 * np.pi * mesh_freq * t)
            # Add harmonics
            signal_clean += 0.5 * np.sin(2 * np.pi * 2 * mesh_freq * t)
            signal_clean += 0.3 * np.sin(2 * np.pi * 3 * mesh_freq * t)

        elif signal_type == "imbalance":
            # Imbalance: 1x RPM component
            rpm = 1500  # RPM
            rotation_freq = rpm / 60.0  # Hz
            signal_clean = np.sin(2 * np.pi * rotation_freq * t)

        else:  # "normal"
            # Normal signal: broadband noise only
            signal_clean = rng.standard_normal(len(t)) * 0.1

        # Add noise
        noise = rng.standard_normal(len(t)) * noise_level
        signal_data = signal_clean + noise

        # Save the signal
        filename = f"test_{signal_type}_{int(sampling_rate)}Hz.csv"
        filepath = DATA_DIR / filename

        # Ensure directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        pd.DataFrame(signal_data, columns=["amplitude"]).to_csv(filepath, index=False, header=False)

        if ctx:
            await ctx.info(f"Test signal saved to {filename}")
            await ctx.info(f"Signal type: {signal_type}, Duration: {duration}s, Fs: {sampling_rate}Hz")

        return f"Successfully generated test signal: {filename}"

    @mcp.tool()
    async def load_signal(
        ctx: Context,
        filepath: str,
        signal_id: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        signal_unit: Optional[Literal["g", "m/s2", "mm/s", "m/s"]] = None,
    ) -> StoredSignalInfo:
        """Load a signal into the in-memory repository for fast repeated access.

        Once loaded, reference the signal by its signal_id in other tools like
        compute_power_spectral_density, diagnose_vibration, etc.

        Signal unit discipline: ISO 20816-3 severity verdicts require a
        DECLARED unit — either via this parameter or a 'signal_unit' field in
        the companion _metadata.json (explicit parameter wins). Units are
        never guessed from signal amplitude; without a declared unit the ISO
        severity block is refused with a structured reason and remedy.

        Args:
            filepath: Filename relative to data/signals/, or absolute path.
            signal_id: Custom ID (defaults to filename stem).
            sampling_rate: Sampling rate in Hz (overrides metadata file).
            signal_unit: Declared signal unit — 'g' or 'm/s2' (acceleration),
                'mm/s' or 'm/s' (velocity). Overrides the metadata file.

        Raises:
            ValueError: If signal_unit is not one of the valid units, or the
                signal data cannot be loaded.
        """
        repo = get_repository()
        info = repo.load_signal(
            filepath,
            signal_id=signal_id,
            sampling_rate=sampling_rate,
            signal_unit=signal_unit,
        )
        if ctx:
            await ctx.info(f"Loaded signal '{info['signal_id']}': {info['num_samples']} samples, {info['size_bytes'] / 1024:.1f} KB")
            if info.get("signal_unit"):
                await ctx.info(f"Signal unit: '{info['signal_unit']}' (declared)")
            else:
                await ctx.info(
                    "Signal unit: not declared — ISO severity verdicts will be "
                    "refused until the unit is declared "
                    "(load_signal(signal_unit=...) or metadata 'signal_unit')."
                )
        return StoredSignalInfo(**info)

    @mcp.tool()
    async def list_stored_signals(ctx: Context) -> list[StoredSignalInfo]:
        """List all signals currently cached in the in-memory repository."""
        repo = get_repository()
        signals = repo.list_signals()
        if ctx:
            await ctx.info(f"{len(signals)} signal(s) in repository")
        return [StoredSignalInfo(**s) for s in signals]

    @mcp.tool()
    async def get_signal_info(ctx: Context, signal_id: str) -> StoredSignalInfo:
        """Get metadata for a stored signal without loading the full array."""
        repo = get_repository()
        info = repo.get_signal_info(signal_id)
        return StoredSignalInfo(**info)

    @mcp.tool()
    async def clear_signal(ctx: Context, signal_id: str) -> dict[str, Any]:
        """Remove a signal from the in-memory repository."""
        repo = get_repository()
        removed = repo.clear_signal(signal_id)
        status = "removed" if removed else "not found"
        if ctx:
            await ctx.info(f"Signal '{signal_id}': {status}")
        return {"signal_id": signal_id, "status": status}

    @mcp.tool()
    async def clear_all_signals(ctx: Context) -> dict[str, Any]:
        """Clear all signals from the in-memory repository."""
        repo = get_repository()
        count = repo.clear_all()
        if ctx:
            await ctx.info(f"Cleared {count} signal(s) from repository")
        return {"cleared_count": count}

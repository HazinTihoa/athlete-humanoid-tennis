"""Launch-condition controls for the MuJoCo tennis trajectory demo."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

LaunchCallback = Callable[[np.ndarray, np.ndarray, float, float], None]
PositionCallback = Callable[[np.ndarray], None]
ResetCallback = Callable[[], None]


def velocity_from_speed_angles(
  speed: float,
  azimuth_deg: float,
  elevation_deg: float,
) -> np.ndarray:
  """Convert speed and court-frame direction angles to a velocity vector."""
  if speed < 0.0:
    raise ValueError("speed must be non-negative")
  azimuth = np.deg2rad(azimuth_deg)
  elevation = np.deg2rad(elevation_deg)
  horizontal_speed = speed * np.cos(elevation)
  return np.array(
    [
      horizontal_speed * np.cos(azimuth),
      horizontal_speed * np.sin(azimuth),
      speed * np.sin(elevation),
    ],
    dtype=np.float64,
  )


def speed_angles_from_velocity(velocity: np.ndarray) -> tuple[float, float, float]:
  """Return speed, XY azimuth, and elevation from a velocity vector."""
  velocity = np.asarray(velocity, dtype=np.float64)
  if velocity.shape != (3,):
    raise ValueError(f"velocity must have shape (3,), got {velocity.shape}")
  speed = float(np.linalg.norm(velocity))
  if speed <= 1e-12:
    return 0.0, 0.0, 0.0
  azimuth = float(np.rad2deg(np.arctan2(velocity[1], velocity[0])))
  elevation = float(
    np.rad2deg(np.arctan2(velocity[2], np.hypot(velocity[0], velocity[1])))
  )
  return speed, azimuth, elevation


class LaunchControlPanel:
  """Small Tk control panel for physical ball launch conditions."""

  UI_SCALE = 2.0

  def __init__(
    self,
    position: np.ndarray,
    velocity: np.ndarray,
    on_launch: LaunchCallback,
    ball_radius: float,
    ball_mass_kg: float,
    position_noise_std: float,
    on_reset_robot: ResetCallback | None = None,
    on_position_change: PositionCallback | None = None,
  ) -> None:
    import tkinter as tk
    from tkinter import ttk

    self._tk = tk
    self._on_launch = on_launch
    self._on_reset_robot = on_reset_robot
    self._on_position_change = on_position_change
    self.ball_radius = float(ball_radius)
    self.closed = False
    speed, azimuth, elevation = speed_angles_from_velocity(velocity)
    self._defaults = (
      np.asarray(position, dtype=np.float64).copy(),
      speed,
      azimuth,
      elevation,
      float(ball_mass_kg),
      float(position_noise_std),
    )

    self.root = tk.Tk()
    tk_scale = float(self.root.tk.call("tk", "scaling"))
    self.root.tk.call("tk", "scaling", tk_scale * self.UI_SCALE)
    self.root.title("Tennis launch controls")
    screen_width = self.root.winfo_screenwidth()
    screen_height = self.root.winfo_screenheight()
    window_width = min(round(440 * self.UI_SCALE), screen_width - 48)
    window_height = min(round(700 * self.UI_SCALE), screen_height - 96)
    self.root.geometry(f"{window_width}x{window_height}+24+48")
    self.root.minsize(min(640, window_width), min(600, window_height))
    self.root.protocol("WM_DELETE_WINDOW", self.close)
    self.root.columnconfigure(0, weight=1)
    self.root.rowconfigure(0, weight=1)

    style = ttk.Style(self.root)
    style.configure("LaunchTitle.TLabel", font=("Sans", 13, "bold"))
    style.configure("LaunchValue.TLabel", font=("Monospace", 9))
    style.configure("LaunchPrimary.TButton", font=("Sans", 10, "bold"))

    self.scroll_canvas = tk.Canvas(
      self.root,
      borderwidth=0,
      highlightthickness=0,
      background=style.lookup("TFrame", "background"),
    )
    scrollbar = ttk.Scrollbar(
      self.root, orient="vertical", command=self.scroll_canvas.yview
    )
    self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
    self.scroll_canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    main = ttk.Frame(self.scroll_canvas, padding=28)
    self._scroll_window = self.scroll_canvas.create_window(
      (0, 0), window=main, anchor="nw"
    )
    main.columnconfigure(0, weight=1)
    main.bind(
      "<Configure>",
      lambda _event: self.scroll_canvas.configure(
        scrollregion=self.scroll_canvas.bbox("all")
      ),
    )
    self.scroll_canvas.bind(
      "<Configure>",
      lambda event: self.scroll_canvas.itemconfigure(
        self._scroll_window, width=event.width
      ),
    )
    self.scroll_canvas.bind_all("<MouseWheel>", self._scroll_with_mousewheel)
    self.scroll_canvas.bind_all("<Button-4>", self._scroll_with_mousewheel)
    self.scroll_canvas.bind_all("<Button-5>", self._scroll_with_mousewheel)

    ttk.Label(main, text="Ball launch", style="LaunchTitle.TLabel").grid(
      row=0, column=0, sticky="w"
    )
    ttk.Label(
      main,
      text="0 deg azimuth points across the net (+X).",
    ).grid(row=1, column=0, sticky="w", pady=(2, 12))

    position_frame = ttk.LabelFrame(main, text="Initial position (m)", padding=10)
    position_frame.grid(row=2, column=0, sticky="ew")
    position_frame.columnconfigure(1, weight=1)
    self.position_vars = [
      tk.DoubleVar(value=float(position[0])),
      tk.DoubleVar(value=float(position[1])),
      tk.DoubleVar(value=float(position[2])),
    ]
    position_limits = (
      ("X", -11.8, 11.8, 0.1),
      ("Y", -5.4, 5.4, 0.1),
      ("Z", self.ball_radius, 5.0, 0.05),
    )
    for row, (label, lower, upper, increment) in enumerate(position_limits):
      ttk.Label(position_frame, text=label, width=4).grid(
        row=row, column=0, sticky="w", pady=3
      )
      ttk.Scale(
        position_frame,
        from_=lower,
        to=upper,
        variable=self.position_vars[row],
      ).grid(row=row, column=1, sticky="ew", padx=(4, 8), pady=3)
      ttk.Spinbox(
        position_frame,
        from_=lower,
        to=upper,
        increment=increment,
        textvariable=self.position_vars[row],
        width=8,
      ).grid(row=row, column=2, sticky="e", pady=3)
      ttk.Label(position_frame, text="m", width=2).grid(
        row=row, column=3, sticky="w", padx=(5, 0), pady=3
      )

    direction_frame = ttk.LabelFrame(main, text="Speed and direction", padding=10)
    direction_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))
    direction_frame.columnconfigure(1, weight=1)
    self.speed_var = tk.DoubleVar(value=speed)
    self.azimuth_var = tk.DoubleVar(value=azimuth)
    self.elevation_var = tk.DoubleVar(value=elevation)
    controls = (
      ("Speed", self.speed_var, 0.0, 40.0, "m/s"),
      ("Azimuth", self.azimuth_var, -180.0, 180.0, "deg"),
      ("Elevation", self.elevation_var, -30.0, 85.0, "deg"),
    )
    for row, (label, variable, lower, upper, unit) in enumerate(controls):
      ttk.Label(direction_frame, text=label, width=10).grid(
        row=row, column=0, sticky="w", pady=4
      )
      ttk.Scale(
        direction_frame,
        from_=lower,
        to=upper,
        variable=variable,
        command=self._refresh_readout,
      ).grid(row=row, column=1, sticky="ew", padx=(4, 8), pady=4)
      ttk.Spinbox(
        direction_frame,
        from_=lower,
        to=upper,
        increment=0.1,
        textvariable=variable,
        width=8,
        command=self._refresh_readout,
      ).grid(row=row, column=2, sticky="e", pady=4)
      ttk.Label(direction_frame, text=unit, width=4).grid(
        row=row, column=3, sticky="w", padx=(5, 0), pady=4
      )

    physical_frame = ttk.LabelFrame(main, text="Ball and estimator", padding=10)
    physical_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
    physical_frame.columnconfigure(1, weight=1)
    self.mass_g_var = tk.DoubleVar(value=1000.0 * ball_mass_kg)
    self.noise_std_var = tk.DoubleVar(value=position_noise_std)
    physical_controls = (
      ("Mass", self.mass_g_var, 10.0, 250.0, 0.5, "g"),
      ("Position noise", self.noise_std_var, 0.0, 0.5, 0.005, "m"),
    )
    for row, (label, variable, lower, upper, increment, unit) in enumerate(
      physical_controls
    ):
      ttk.Label(physical_frame, text=label, width=14).grid(
        row=row, column=0, sticky="w", pady=4
      )
      ttk.Scale(
        physical_frame,
        from_=lower,
        to=upper,
        variable=variable,
        command=self._refresh_readout,
      ).grid(row=row, column=1, sticky="ew", padx=(4, 8), pady=4)
      ttk.Spinbox(
        physical_frame,
        from_=lower,
        to=upper,
        increment=increment,
        textvariable=variable,
        width=8,
        command=self._refresh_readout,
      ).grid(row=row, column=2, sticky="e", pady=4)
      ttk.Label(physical_frame, text=unit, width=4).grid(
        row=row, column=3, sticky="w", padx=(5, 0), pady=4
      )

    self.velocity_text = tk.StringVar()
    ttk.Label(
      main,
      textvariable=self.velocity_text,
      style="LaunchValue.TLabel",
      wraplength=380,
    ).grid(row=5, column=0, sticky="w", pady=(12, 0))

    self.status_text = tk.StringVar(value="Adjust values, then press Launch.")
    ttk.Label(main, textvariable=self.status_text, wraplength=380).grid(
      row=6, column=0, sticky="w", pady=(5, 12)
    )

    buttons = ttk.Frame(main)
    buttons.grid(row=7, column=0, sticky="ew")
    buttons.columnconfigure(0, weight=1)
    buttons.columnconfigure(1, weight=1)
    ttk.Button(
      buttons,
      text="Launch",
      command=self._launch,
      style="LaunchPrimary.TButton",
    ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
    ttk.Button(buttons, text="Defaults", command=self._restore_defaults).grid(
      row=0, column=1, sticky="ew", padx=(5, 0)
    )
    if self._on_reset_robot is not None:
      ttk.Button(
        buttons,
        text="Reset G1",
        command=self._reset_robot,
      ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    for variable in self.position_vars:
      variable.trace_add("write", self._position_changed)
    for variable in (
      self.speed_var,
      self.azimuth_var,
      self.elevation_var,
      self.mass_g_var,
      self.noise_std_var,
    ):
      variable.trace_add("write", self._refresh_readout)
    self.root.bind("<Return>", lambda _event: self._launch())
    self._refresh_readout()

  def _position(self) -> np.ndarray:
    position = np.array(
      [variable.get() for variable in self.position_vars], dtype=np.float64
    )
    if not np.all(np.isfinite(position)):
      raise ValueError("position must contain finite values")
    if position[2] < self.ball_radius:
      raise ValueError(
        f"Z must be at least the ball radius ({self.ball_radius:.4f} m)"
      )
    return position

  def _values(
    self,
  ) -> tuple[np.ndarray, float, float, float, float, float]:
    position = self._position()
    speed = float(self.speed_var.get())
    azimuth = float(self.azimuth_var.get())
    elevation = float(self.elevation_var.get())
    ball_mass_kg = 0.001 * float(self.mass_g_var.get())
    position_noise_std = float(self.noise_std_var.get())
    if not np.isfinite(
      [speed, azimuth, elevation, ball_mass_kg, position_noise_std]
    ).all():
      raise ValueError("launch settings must contain finite values")
    if speed < 0.0:
      raise ValueError("speed must be non-negative")
    if ball_mass_kg <= 0.0:
      raise ValueError("ball mass must be positive")
    if position_noise_std < 0.0:
      raise ValueError("position noise standard deviation must be non-negative")
    return (
      position,
      speed,
      azimuth,
      elevation,
      ball_mass_kg,
      position_noise_std,
    )

  def _refresh_readout(self, *_args) -> None:
    try:
      (
        _position,
        speed,
        azimuth,
        elevation,
        ball_mass_kg,
        position_noise_std,
      ) = self._values()
      velocity = velocity_from_speed_angles(speed, azimuth, elevation)
      self.velocity_text.set(
        f"Velocity xyz = ({velocity[0]:6.2f}, {velocity[1]:6.2f}, "
        f"{velocity[2]:6.2f}) m/s\n"
        f"Mass = {1000.0 * ball_mass_kg:.1f} g, "
        f"position noise sigma = {position_noise_std:.3f} m"
      )
    except (ValueError, self._tk.TclError):
      self.velocity_text.set("Launch settings contain invalid input")

  def _position_changed(self, *_args) -> None:
    self._refresh_readout()
    if self._on_position_change is None:
      return
    try:
      position = self._position()
      self._on_position_change(position)
      self.status_text.set(
        "Launch point: "
        f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m"
      )
    except (ValueError, self._tk.TclError) as exc:
      self.status_text.set(f"Invalid launch position: {exc}")
    except Exception as exc:
      self.status_text.set(f"Could not update launch position: {exc}")

  def _launch(self) -> None:
    try:
      (
        position,
        speed,
        azimuth,
        elevation,
        ball_mass_kg,
        position_noise_std,
      ) = self._values()
      velocity = velocity_from_speed_angles(speed, azimuth, elevation)
      self._on_launch(
        position,
        velocity,
        ball_mass_kg,
        position_noise_std,
      )
      self.status_text.set(
        f"Launch queued: {speed:.2f} m/s, {1000.0 * ball_mass_kg:.1f} g, "
        f"position noise sigma {position_noise_std:.3f} m."
      )
    except (ValueError, self._tk.TclError) as exc:
      self.status_text.set(f"Invalid launch settings: {exc}")

  def _restore_defaults(self) -> None:
    (
      position,
      speed,
      azimuth,
      elevation,
      ball_mass_kg,
      position_noise_std,
    ) = self._defaults
    for variable, value in zip(self.position_vars, position):
      variable.set(float(value))
    self.speed_var.set(speed)
    self.azimuth_var.set(azimuth)
    self.elevation_var.set(elevation)
    self.mass_g_var.set(1000.0 * ball_mass_kg)
    self.noise_std_var.set(position_noise_std)
    self.status_text.set("Defaults restored; launch position preview updated.")

  def _reset_robot(self) -> None:
    if self._on_reset_robot is None:
      return
    try:
      self._on_reset_robot()
      self.status_text.set("G1 and rally reset queued.")
    except Exception as exc:
      self.status_text.set(f"Could not reset G1: {exc}")

  def poll(self) -> bool:
    if self.closed:
      return False
    try:
      self.root.update_idletasks()
      self.root.update()
    except self._tk.TclError:
      self.closed = True
    return not self.closed

  def _scroll_with_mousewheel(self, event) -> None:
    if getattr(event, "num", None) == 4:
      delta = -1
    elif getattr(event, "num", None) == 5:
      delta = 1
    else:
      delta = -int(getattr(event, "delta", 0) / 120)
    if delta:
      self.scroll_canvas.yview_scroll(delta, "units")

  def close(self) -> None:
    if self.closed:
      return
    self.closed = True
    try:
      self.scroll_canvas.unbind_all("<MouseWheel>")
      self.scroll_canvas.unbind_all("<Button-4>")
      self.scroll_canvas.unbind_all("<Button-5>")
      self.root.destroy()
    except self._tk.TclError:
      pass

import av
import json
import argparse
import numpy as np

from typing import Dict, List, Tuple, Union
from fractions import Fraction
from logging import Logger
from pathlib import Path

from .utils import NULL_LOGGER, Real
from .edit_context import TimelineChange, EditCtx, VideoEditContext, AudioEditContext

ID_VIDEO_STREAM = 'video'  # compatible with PyAV
ID_AUDIO_STREAM = 'audio'  # compatible with PyAV
SUPPORTED_STREAM_TYPES = [ID_VIDEO_STREAM, ID_AUDIO_STREAM]

ID_CODEC = 'codec'
ID_CODEC_OPTIONS = 'codec-options'
ID_BITRATE = 'bitrate'
ID_RESOLUTION = 'resolution'
ID_MAX_FPS = 'max-fps'
ID_SAMPLE_RATE = 'sample-rate'
ID_MONO = 'mono'
ID_TIMELINE_CHANGES = 'timeline-changes'


class Editor:

  def __init__(self,
               video_settings: dict = None,
               audio_settings: dict = None,
               logger: Logger = NULL_LOGGER):
    """Edits and exports recordings according to the configuration

    Args:
        video_settings (dict, optional): General video settings. Defaults to None.
        audio_settings (dict, optional): General audio settings. Defaults to None.
        logger (Logger, optional): Logger for messages. Defaults to NULL_LOGGER.
    """
    self.logger = logger
    self.settings = {}
    self.settings[ID_VIDEO_STREAM] = {} if video_settings is None else video_settings
    self.settings[ID_AUDIO_STREAM] = {} if audio_settings is None else audio_settings

  def edit(self, src_path: str, changes: Union[List[TimelineChange], np.ndarray], dst_path: str) \
    -> None:
    """Edits and exports a recording according to a provided timeline changes and current editor
    settings

    Args:
        src_path (str): Path of the original recording
        changes (Union[List[TimelineChange], np.ndarray]): Timeline changes to be made - a list of
        TimelineChange instances or a numpy array of shape (N, 3), where N is the number of changes
        dst_path (str): Path of the output recording
    """
    self.logger.info(f'Started editing: "{src_path}"')

    src_path = str(Path(src_path).resolve())
    changes = TimelineChange.from_numpy(changes) if isinstance(changes, np.ndarray) else changes
    dst_path = str(Path(dst_path).resolve())
    source = av.open(str(src_path))
    dest, ctx_map = self.prepare_destination(source, dst_path)

    # find DTS of first packets to know which one to seek for begining
    first_pkts = {}
    for idx, ctx in ctx_map.items():
      ctx.seek_beginning()
      first_pkt = next(source.demux(ctx.src_stream))
      first_pkts[idx] = Real(first_pkt.dts * first_pkt.time_base)
    streams_ordered = [k for k, v in sorted(first_pkts.items(), key=lambda kv: kv[1])]

    # prepare contexts of streams for editing
    valid_streams = {}
    for idx, ctx in ctx_map.items():
      if ctx.prepare_for_editing(changes):
        valid_streams[idx] = ctx.src_stream

    # edit
    if len(valid_streams) > 0:
      first_ctx = [idx for idx in streams_ordered if idx in valid_streams][0]
      ctx_map[first_ctx].seek_beginning()

      for src_packet in source.demux(list(valid_streams.values())):
        ctx = ctx_map[src_packet.stream.index]
        if ctx.is_done:
          continue
        for dst_packet in ctx.decode_edit_encode(src_packet):
          dest.mux(dst_packet)
        if all(ctx.is_done for ctx in ctx_map.values()):  # early stop when all are done
          break

      # flush buffers
      for dst_stream in dest.streams:
        dest.mux(dst_stream.encode())
    else:
      self.logger.warning(f'Editing: "{src_path}" resulted in an empty recording')
    self.logger.info(f'Finished editing: "{src_path}" -> "{dst_path}"')

    dest.close()
    source.close()

  def prepare_destination(self, source: av.container.InputContainer, dst_path: str) \
    -> Tuple[av.container.OutputContainer, Dict[int, EditCtx]]:
    """Prepares a destination container to hold the edited recording

    Args:
        source (av.container.InputContainer): Container of the original recording
        dst_path (str): Path of the output recording

    Returns:
        Tuple[av.container.OutputContainer, Dict[int, EditCtx]]: Destination container and a mapping
        from its streams to editing contexts
    """
    dst = av.open(dst_path, mode='w')
    ctx_map = {}

    valid_streams = []
    for stream in source.streams:
      if stream.type in SUPPORTED_STREAM_TYPES:
        if stream.codec_context is not None:
          valid_streams.append(stream)
        else:
          self.logger.warning(f'Skipping #{stream.index} stream (no decoder available)')
      else:
        self.logger.warning(f'Skipping #{stream.index} stream ({stream.type} not supported)')

    for src_stream in valid_streams:
      # stream-specific settings take precedence over type-specific settings
      settings = self.settings.get(src_stream.type, {}).copy()
      settings.update(self.settings.get(src_stream.index, {}))

      if src_stream.type == ID_VIDEO_STREAM:
        codec = settings.get(ID_CODEC, src_stream.codec_context.name)
        codec_options = settings.get(ID_CODEC_OPTIONS, src_stream.codec_context.options)
        bitrate = settings.get(ID_BITRATE, src_stream.bit_rate)
        resolution = settings.get(ID_RESOLUTION, [src_stream.width, src_stream.height])
        max_fps = Fraction(settings.get(ID_MAX_FPS, src_stream.guessed_rate))

        dst_stream = dst.add_stream(codec_name=codec, options=codec_options)
        dst_stream.codec_context.time_base = Fraction(1, 60000)
        dst_stream.time_base = Fraction(1, 60000)  # might not work
        dst_stream.pix_fmt = src_stream.pix_fmt
        dst_stream.bit_rate = bitrate
        dst_stream.width, dst_stream.height = resolution
        ctx_map[src_stream.index] = VideoEditContext(src_stream, dst_stream, max_fps)

      elif src_stream.type == ID_AUDIO_STREAM:
        codec = settings.get(ID_CODEC, src_stream.codec_context.name)
        codec_options = settings.get(ID_CODEC_OPTIONS, src_stream.codec_context.options)
        bitrate = settings.get(ID_BITRATE, src_stream.bit_rate)
        sample_rate = settings.get(ID_SAMPLE_RATE, src_stream.sample_rate)
        channels = 1 if settings.get(ID_MONO, False) else src_stream.channels

        dst_stream = dst.add_stream(codec_name=codec, rate=sample_rate)
        dst_stream.options = codec_options
        dst_stream.bit_rate = bitrate
        dst_stream.channels = channels
        ctx_map[src_stream.index] = AudioEditContext(src_stream, dst_stream)

      src_stream.thread_type = 'AUTO'
      dst_stream.thread_type = 'AUTO'

    # check if all video streams have high enough resolution of the time_base
    dst.start_encoding()
    for vid_ctx in [c for c in ctx_map.values() if isinstance(c, VideoEditContext)]:
      possible_fps = 1 / vid_ctx.dst_stream.time_base
      if possible_fps < vid_ctx.max_fps:
        self.logger.warning(f'Low time base resolution of #{dst_stream.index} video stream - '
                            f'maxfps must be limited from {vid_ctx.max_fps} to {possible_fps}')
        vid_ctx.max_fps = possible_fps

    return dst, ctx_map

  def export_json(self, path, changes: Union[List[TimelineChange], np.ndarray] = None) -> None:
    """Exports the current configuration of the editor with a list of timeline changes (if provided)

    Args:
        path ([type]): Path of the output json file
        changes (Union[List[TimelineChange], np.ndarray]): Timeline changes - a list of
        TimelineChange instances or a numpy array of shape (N, 3), where N is the number of changes
    """
    assert ID_TIMELINE_CHANGES not in self.settings

    with open(path, 'w', encoding='UTF-8') as fp:
      config = self.settings.copy()
      if isinstance(changes, np.ndarray):
        config[ID_TIMELINE_CHANGES] = changes.tolist()
      elif changes is not None:
        config[ID_TIMELINE_CHANGES] = [[r.beg, r.end, r.multi] for r in changes]
      json.dump(config, fp)

  @staticmethod
  def from_json(json_settings: dict, logger: Logger = NULL_LOGGER) -> 'Editor':
    """Constructs an Editor from a dictionary of settings.

    Returns:
        Editor: Configured editor prepared for editing
    """
    editor = Editor(logger=logger)
    for identifier, config in json_settings.items():
      identifier = identifier.lower()

      if identifier in [ID_VIDEO_STREAM, ID_AUDIO_STREAM]:
        settings = editor.settings.setdefault(identifier, {})  # stream type
      elif identifier.isnumeric():
        settings = editor.settings.setdefault(int(identifier), {})  # stream idx
      else:
        if identifier != ID_TIMELINE_CHANGES:
          logger.warning(f'Skipping unrecognized identifier: {identifier}')
        continue

      for key, value in config.items():
        if key == ID_CODEC:
          settings[key] = str(value)
        elif key == ID_CODEC_OPTIONS:
          settings[key] = value
          for option_key, option_value in value.items():
            value[option_key] = str(option_value)
        elif key == ID_BITRATE:
          settings[key] = int(value)  # bitrate in b/s
          if settings[key] <= 0:
            raise ValueError(f'"{ID_BITRATE}" must be a positive number')
        elif key == ID_RESOLUTION:
          settings[key] = [int(dim) for dim in value]  # [width, height]
          if settings[key][0] * settings[key][1] <= 0:
            raise ValueError(f'"{ID_RESOLUTION}" must consist of positive numbers')
        elif key == ID_MAX_FPS:
          settings[key] = float(value)
          if settings[key] <= 0:
            raise ValueError(f'"{ID_MAX_FPS}" must be a positive number')
        elif key == ID_SAMPLE_RATE:
          settings[key] = int(value)
          if settings[key] <= 0:
            raise ValueError(f'"{ID_SAMPLE_RATE}" must be a positive number')
        elif key == ID_MONO:
          settings[key] = bool(value)
        else:
          logger.warning(f'Skipping unrecognized setting: {key}:')
    return editor


############################################### CLI ################################################

NAME = 'editor'
DESCRIPTION = 'Edits recordings according to the configuration'
ARG_SRC = 'src'
ARG_DST = 'dst'
DEFAULT_ARGS = {}


def setup_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
  """Sets up a CLI argument parser for this submodule

  Returns:
      argparse.ArgumentParser: Configured parser
  """
  parser.description = DESCRIPTION
  parser.add_argument(ARG_SRC, help='Path of the recording to edit', type=Path, action='store')
  parser.add_argument(ARG_DST, help='Path of the edited recording', type=Path, action='store')
  parser.set_defaults(run=run_submodule)
  # TODO
  return parser


def run_submodule(args: object, logger: Logger) -> None:
  """Runs this submodule

  Args:
      args (object): Arguments of this submodule (defined in setup_arg_parser)
      logger (Logger): Logger for messages
  """
  # TODO
  args = args.__dict__
  with open('test.json', 'r', encoding='UTF-8') as fp:
    json_cfg = json.load(fp)
  editor = Editor.from_json(json_cfg, logger=logger)
  changes = np.array(json_cfg[ID_TIMELINE_CHANGES])
  # editor.export_json(changes, 'test2.json')
  editor.edit(args[ARG_SRC], changes, args[ARG_DST])

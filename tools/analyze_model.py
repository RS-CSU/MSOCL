# -*- coding: utf-8 -*-
# noqa: B950

import logging
from collections import Counter
import tqdm
import os

from fsdet.checkpoint import DetectionCheckpointer
from fsdet.config import get_cfg
from fsdet.data import build_detection_test_loader
from fsdet.engine import default_argument_parser
from fsdet.modeling import build_model
from fsdet.utils.analysis import (
    activation_count_operators,
    flop_count_operators,
    parameter_count_table,
)
from fsdet.utils.logger import setup_logger

logger = logging.getLogger("fsdet")


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    setup_logger()
    return cfg


def do_flop(cfg):
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    counts = Counter()
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        counts += flop_count_operators(model, data)
    logger.info(
        "(G)Flops for Each Type of Operators:\n" + str([(k, v / idx) for k, v in counts.items()])
    )


def do_activation(cfg):
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    counts = Counter()
    for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
        counts += activation_count_operators(model, data)
    logger.info(
        "(Million) Activations for Each Type of Operators:\n"
        + str([(k, v / idx) for k, v in counts.items()])
    )


def do_parameter(cfg):
    model = build_model(cfg)
    logger.info("Parameter Count:\n" + parameter_count_table(model, max_depth=5))


def do_structure(cfg):
    model = build_model(cfg)
    logger.info("Model Structure:\n" + str(model))


if __name__ == "__main__":
    os.chdir('../')
    parser = default_argument_parser()
    parser.add_argument(
        "--tasks",
        choices=["flop", "activation", "parameter", "structure"],
        default="parameter",
        # required=True,
        nargs="+",
    )
    parser.add_argument(
        "--num-inputs",
        default=100,
        type=int,
        help="number of inputs used to compute statistics for flops/activations, "
        "both are data dependent.",
    )
    args = parser.parse_args()
    assert not args.eval_only
    assert args.num_gpus == 1

    cfg = setup(args)

    for task in [args.tasks]:
        {
            "flop": do_flop,
            "activation": do_activation,
            "parameter": do_parameter,
            "structure": do_structure,
        }[task](cfg)

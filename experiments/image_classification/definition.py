from copy import deepcopy

import numpy as np

WORKERS = 25


def get_eval():
  return {
    "training": {
      "rcond": 1e-15,
    },
    "runtime": {
      "workers": WORKERS,
    },
  }


def get_postprocess():
  return {
    "detector": {
      "darkcounts": 0,
      "detector_efficiency": 0.9,
    },
    "runtime": {
      "workers": WORKERS,
      "split_factor": 1,
    },
    "sampling": {
      "seed": 1234,
      "f": 10_000_000,
    },
    "renorm": {
      "remove_dummy": False,
      "pthresh": 1e-3,
    },
    "postsel": {
      "lower": 1,
      "upper": 4,
    },
  }


def get_experiments():
  quantum_states = [
    {
      "type": "polarised",
      "params": {
        "state": {
          "type": "fock",
          "params": {
            "occupancies": [1, 1, 1, 1, 0],
          },
        },
        "axis": 0,
      },
    },
  ]
  semiclassical_states = [
    {
      "type": "polarised",
      "params": {
        "state": {
          "type": "coherent",
          "params": {"alphas": [1.5, 1.5, 0, 0, 0], "truncation": 6},
        },
        "axis": 0,
      },
    },
  ]
  classical_states = [
    {
      "type": "polarised",
      "params": {
        "state": {
          "type": "classical_coherent",
          "params": {"alphas": [1.5, 1.5, 0, 0, 0]},
        },
        "axis": 0,
      },
    },
  ]

  encodings = [
    {
      "type": "compose",
      "params": {
        "encodings": [
          {
            "type": "n_pad_end",
            "params": {
              "value": 0,
            },
          },
          {
            "type": "linear_match_ranges",
            "params": {
              "in_min": 0.0,
              "in_max": 1.0,
              "out_min": np.pi / 4,
              "out_max": 3 * np.pi / 4,
              "name": "linear_scale",
            },
          },
        ]
      },
    },
  ]
  reservoirs = [
    {
      "type": "stack",
      "params": {
        "reservoirs": [
          {
            "type": "mesh",
            "params": {
              "seed": 234583478569823487239 + seed,
              "num_spatial_modes": 5,
              "depth": 7,
              "polarised": True,
              "lossy": False,
              "coupling_fn": "qhp_qhp_(theta_2x2)_all_uniform",
            },
          },
          {
            "type": "mesh",
            "params": {
              "seed": 2345834785698234872396 + seed,
              "num_spatial_modes": 5,
              "depth": 10,  # TBD
              "polarised": True,
              "lossy": False,
              "coupling_fn": "qhp_qhp_(theta_2x2)_parameterised",
            },
          },
          {
            "type": "mesh",
            "params": {
              "seed": 23458347856982348723966 + seed,
              "num_spatial_modes": 5,
              "depth": 7,
              "polarised": True,
              "lossy": False,
              "coupling_fn": "qhp_qhp_(theta_2x2)_all_uniform",
            },
          },
        ],
        "offsets": [0, 0, 0],
      },
    }
    for seed in range(1)
  ]

  _ds_meta = {
    "postfilters": [
      {
        "type": "subsample",
        "params": {
          "n_train": 10_000,
          "n_test": 1000,
          "strategy": "random",
          "seed": 763237,
        },
      },
      {
        "type": "pca",
        "params": {
          "n_components": 20,
        },
      },
      {
        "type": "classification",
        "params": {},
      },
    ],
    "tasks": [
      {
        "type": "classify_one_hot",
        "params": {},
      },
    ],
  }

  datasets = [
    {
      "type": "func",
      "params": {
        "key": "mnist_v1",
        **deepcopy(_ds_meta),
      },
    },
    {
      "type": "func",
      "params": {
        "key": "breast_v1",
        **deepcopy(_ds_meta),
      },
    },
    {
      "type": "func",
      "params": {
        "key": "blood_v1",
        **deepcopy(_ds_meta),
      },
    },
    {
      "type": "func",
      "params": {
        "key": "derma_v1",
        **deepcopy(_ds_meta),
      },
    },
    {
      "type": "func",
      "params": {
        "key": "path_v1",
        **deepcopy(_ds_meta),
      },
    },
    {
      "type": "func",
      "params": {
        "key": "organc_v1",
        **deepcopy(_ds_meta),
      },
    },
    {
      "type": "func",
      "params": {
        "key": "oct_v1",
        **deepcopy(_ds_meta),
      },
    },
  ]

  # Update the seeds.
  # This is purely for reproducibility, can use whichever seeds you want.
  # seeds = [763235, 763236, 763237, 763238, 763235, 763236, 763237]
  # for d, s in zip(datasets, seeds):
  #   d["params"]["postfilters"][0]["params"]["seed"] = s

  tensor_experiments_defn = [
    {
      "name": "im_class_quantum",
      "simulator": {
        "type": "SLOS_probs",
        "params": {"workers": WORKERS, "split_factor": 100},
      },
      "states": quantum_states,
      "datasets": datasets,
      "encodings": encodings,
      "reservoirs": reservoirs,
    },
    {
      "name": "im_class_semiclassical",
      "simulator": {
        "type": "Efficient_quantum_coherent",
        "params": {
          "workers": WORKERS,
          "split_factor": 50,
          "min_prob": 10**-8,
        },
      },
      "states": semiclassical_states,
      "datasets": datasets,
      "encodings": encodings,
      "reservoirs": reservoirs,
    },
    {
      "name": "im_class_classical",
      "simulator": {
        "type": "classical",
        "params": {
          "workers": WORKERS,
          "split_factor": 1,
        },
      },
      "states": classical_states,
      "datasets": datasets,
      "encodings": encodings,
      "reservoirs": reservoirs,
    },
  ]

  return tensor_experiments_defn

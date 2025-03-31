def get_postprocess():
  return {
    "detector": {
      "darkcounts": 0.1,
      "detector_efficiency": 0.9,
    },
    "runtime": {
      "workers": 0,
      "split_factor": 1,
    },
    "sampling": {
      "seed": 1234,
      "f": 10_000,
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
            "occupancies": [1, 0, 0],
          },
        },
        "axis": 0,
      },
    },
    {
      "type": "polarised",
      "params": {
        "state": {
          "type": "dist",
          "params": {
            "labels": "{d:0},0,0",
          },
        },
        "axis": 0,
      },
    },
    {
      "type": "polarised",
      "params": {
        "state": {
          "type": "photon_added_coherent",
          "params": {
            "alphas": [0.1, 0, 0],
            "truncation": 3,
            "adds": [1, 0, 0],
          },
        },
        "axis": 0,
        "name": "photon_added_coherent",
      },
    },
  ]
  semiclassical_states = [
    {
      "type": "polarised",
      "params": {
        "state": {
          "type": "coherent",
          "params": {"alphas": [0.1, 0, 0], "truncation": 4},
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
          "params": {"alphas": [0.1, 0, 0]},
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
            "type": "linear_one_to_many",
            "params": {},
          },
          {
            "type": "general_pol_encoding",
            "params": {
              "seed": 1234,  # for phases
              "N": 2,  # period
              "K": 0,  # Not proportional to port
              "P": 1,  # random phases
              "F": 0,  # equator only
              "G": 0,  # Not interleaved
              "name": "uniform_linear_rand_phase",
            },
          },
        ]
      },
    },
    {
      "type": "compose",
      "params": {
        "encodings": [
          {
            "type": "linear_one_to_many",
            "params": {},
          },
          {
            "type": "general_pol_encoding",
            "params": {
              "seed": 1234,  # for phases
              "N": 2,  # period
              "K": 1,  # proportional to port
              "P": 1,  # random phases
              "F": 0,  # equator only
              "G": 0,  # Not interleaved
              "name": "multilinear_rand_phase",
            },
          },
        ]
      },
    },
    {
      "type": "compose",
      "params": {
        "encodings": [
          {
            "type": "linear_one_to_many",
            "params": {},
          },
          {
            "type": "general_pol_encoding",
            "params": {
              "seed": 1234,  # for phases
              "N": 4,  # period
              "K": 1,  # proportional to port
              "P": 1,  # random phases
              "F": 1,  # spiral
              "G": 0,  # Not interleaved
              "name": "multispiral_overlap_rand_phase",
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
            "type": "layer",
            "params": {
              "seed": None,  # Not used, coupling is deterministic in data
              "num_spatial_modes": 3,
              "polarised": True,
              "lossy": False,
              "coupling_fn": "qh_1x1",
            },
          },
          {
            "type": "tri",
            "params": {
              "seed": 234583478569823487239 + seed,
              "num_spatial_modes": 3,
              "polarised": True,
              "lossy": False,
              "coupling_fn": "qhp_qhp_(theta_2x2)_all_uniform",
            },
          },
        ],
        "offsets": [0, 0],
      },
    }
    for seed in range(1)
  ]

  datasets = [
    {
      "type": "func",
      "params": {
        "key": "1xn_linspace_n1-p1_generator",
        "split": 0.5,
        "seed": 7568,
        "n": 10,
        "start": -1,
        "end": 1,
        "endpoint": False,
        "tasks": [
          {
            "type": "sinc",
            "params": {},
          },
          {
            "type": "tanh",
            "params": {},
          },
          {
            "type": "rect",
            "params": {},
          },
          *[
            {
              "type": "random_freq",
              "params": {"seed": 12, "index": index},
            }
            for index in range(32)
          ],
        ],
      },
    },
  ]

  tensor_experiments_defn = [
    {
      "name": "2411_quantum",
      "simulator": {
        "type": "SLOS_probs",
        "params": {"workers": 4, "split_factor": 1},
      },
      "states": quantum_states,
      "datasets": datasets,
      "encodings": encodings,
      "reservoirs": reservoirs,
    },
    {
      "name": "2411_semiclassical",
      "simulator": {
        "type": "Efficient_quantum_coherent",
        "params": {},
      },
      "states": semiclassical_states,
      "datasets": datasets,
      "encodings": encodings,
      "reservoirs": reservoirs,
    },
    {
      "name": "2411_classical",
      "simulator": {
        "type": "classical",
        "params": {},
      },
      "states": classical_states,
      "datasets": datasets,
      "encodings": encodings,
      "reservoirs": reservoirs,
    },
  ]

  return tensor_experiments_defn

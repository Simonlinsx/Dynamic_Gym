window.projectData = {
  updated: "2026-06-18",
  heroStats: [
    { label: "Task families", value: "4 final tasks" },
    { label: "Affordance labels", value: "32 / 32 assets" },
    { label: "Active robot", value: "Franka + Revo2" },
    { label: "Pipeline", value: "Teacher -> Student" },
  ],
  taskSettings: {
    intro:
      "The final Dynamic Gym benchmark is organized around four dynamic dexterous manipulation targets: two aerial object settings and two tabletop object settings. Each task keeps the same high-level policy interface, but changes the object motion source, affordance prior, and terminal manipulation objective.",
    groups: [
      {
        label: "Aerial dynamic tasks",
        value: "2",
        body: "Objects move through free space before or during contact. The policy must predict the catch region, absorb motion, and stabilize the grasp.",
      },
      {
        label: "Tabletop dynamic tasks",
        value: "2",
        body: "Objects move across a surface by rolling, conveyor motion, or cart motion. The policy must intercept without unsafe impact, grasp the correct affordance, and finish with a task-specific pose.",
      },
    ],
    columns: [
      "Setting",
      "Task",
      "Objects",
      "Dynamic source",
      "Affordance design",
      "Final objective",
    ],
    rows: [
      {
        setting: "Aerial",
        settingTone: "aerial",
        task: "Falling Baton (主动抓取)",
        objects: "baton / marker / toy screwdriver",
        dynamics: "Free fall with random angular velocity",
        affordance: "Middle or handle is graspable; both ends are negative",
        objective: "Stick-catching game setting: actively predict, intercept, grasp, and hold the falling baton",
      },
      {
        setting: "Aerial",
        settingTone: "aerial",
        task: "Aerial Object Catch (空中抛落物接住)",
        objects: "rod-like object",
        dynamics: "Airborne toss/drop, guided free fall, or low-speed handoff",
        affordance: "Specified safe grasp region",
        objective: "Wait near the receive zone, absorb impact, close on the safe region, and stabilize the airborne object",
      },
      {
        setting: "Tabletop",
        settingTone: "tabletop",
        task: "Rolling Marker",
        objects: "marker / battery / cylinder",
        dynamics: "Ramp rolling or tabletop rolling",
        affordance: "Tip is negative; body is graspable",
        objective: "Capture the rolling object, align its axis, and place it into a holder",
      },
      {
        setting: "Tabletop",
        settingTone: "tabletop",
        task: "Conveyor Tool",
        objects: "screwdriver / spoon / brush",
        dynamics: "Conveyor belt or moving cart",
        affordance: "Handle graspable; functional end negative",
        objective: "Grasp the functional tool by the handle and correct its pose for downstream use",
      },
    ],
    references: [
      {
        title: "Falling Baton as a stick-catching game",
        src: "assets/images/falling_baton_game_schematic.svg",
        caption:
          "The active aerial task is framed like a stick-catching game: the robot predicts the future baton position, intercepts the safe middle or handle region, and holds the object after contact.",
      },
      {
        title: "Aerial Object Catch reference setting",
        src: "assets/images/v88_falling_baton_screwdriver.jpg",
        caption:
          "Aerial Object Catch replaces the earlier Baton Insert wording. It follows the thrown-object catching spirit of Catch It! while using a fixed Franka + BrainCo Revo2 platform.",
        href: "https://arxiv.org/pdf/2409.10319",
      },
    ],
  },
  pipeline: [
    {
      title: "Task modules",
      body: "Dynamic tabletop and falling-object settings share the same robot interface, reward logging, video capture, and W&B tracking.",
    },
    {
      title: "Privileged teacher",
      body: "The teacher receives simulator object state, clean mesh point clouds, object velocity, contact/lift signals, and clean-v2 affordance labels.",
    },
    {
      title: "Deployable student",
      body: "The student is designed around RGB-D object-mask point-cloud sequences, temporal tracking, point-flow prediction, and object-local affordance prediction.",
    },
    {
      title: "Robot adapter",
      body: "Franka plus Inspire or BrainCo Revo2 are kept as separate generated assets and action adapters. Current Revo2 ablations use an 11D physical hand action space.",
    },
    {
      title: "Evaluation loop",
      body: "Training curves, saved videos, deterministic eval, per-object metrics, and artifact pages are used to decide which modules move forward.",
    },
  ],
  modules: [
    {
      label: "Teacher observation",
      title: "Privileged simulator state",
      body: "Upper-bound training signal for debugging reward, embodiment, and task feasibility before deploying RGB-D observations.",
      items: [
        "Robot joint state, previous targets, palm state, and fingertip positions",
        "Object pose, velocity, clean mesh point cloud, and point-cloud centroid",
        "Clean-v2 grasp affordance labels and contact/lift diagnostics",
      ],
    },
    {
      label: "Teacher policy",
      title: "PPO actor-critic",
      body: "Learns task behavior under privileged information and supplies behavior targets for later student distillation.",
      items: [
        "Reach -> contact -> grasp -> lift/hold reward family",
        "Task-specific catch rewards for Falling Baton V88",
        "W&B curves for reward terms, success, contact, and per-object metrics",
      ],
    },
    {
      label: "Student perception",
      title: "RGB-D temporal point cloud",
      body: "The deployable branch avoids direct object pose in the actor and uses a mask-derived partial point-cloud sequence.",
      items: [
        "Object mask sequence and RGB-D point cloud fusion",
        "Temporal tracking confidence and centroid velocity proxy",
        "Auxiliary point-flow and object-local affordance prediction heads",
      ],
    },
    {
      label: "Student policy",
      title: "Distilled deployable actor",
      body: "Student training will combine PPO with teacher action/value supervision and perception auxiliary losses.",
      items: [
        "Teacher action distillation and optional value/feature distillation",
        "Affordance BCE over valid clean-v2 labels",
        "Point-flow and temporal consistency losses",
      ],
    },
    {
      label: "Safety adapter",
      title: "Real-robot execution layer",
      body: "The final policy output should pass through a rate-limited robot adapter before sim2real trials.",
      items: [
        "Joint and collision limits for Franka + Revo2",
        "Action smoothing and hand target delta penalties",
        "Task-level monitoring for drops, impacts, and unstable contacts",
      ],
    },
  ],
  affordance: {
    intro:
      "The current affordance set covers the full 32-object pool. The recommended training target is the clean binary v2 grasp label: useful as weak reward shaping or auxiliary supervision, not as a hard success condition.",
    metrics: [
      {
        label: "Annotated assets",
        value: "32 / 32",
        detail: "DextoolBench12: 12 / 12, DOMINO20: 20 / 20.",
      },
      {
        label: "Positive-only labels",
        value: "32 files",
        detail: "Each asset has positive_grasp_only.npz with raw, connected, and filled positive masks.",
      },
      {
        label: "Clean binary v2",
        value: "32 files",
        detail: "Recommended file: grasp_affordance_clean_v2.npz.",
      },
      {
        label: "Positive vertices",
        value: "247,536",
        detail: "Clean-v2 grasp_label = 1 across all objects.",
      },
      {
        label: "Negative vertices",
        value: "73,551",
        detail: "Conservative grasp_label = 0 regions.",
      },
      {
        label: "Ignored vertices",
        value: "127,469",
        detail: "Uncertain or conflict regions excluded from BCE and reward.",
      },
    ],
    rawStats: [
      {
        label: "DextoolBench12 clean v2",
        vertices: "12 / 12 assets",
        labeled: "26,844 positive / 27,579 negative",
        overlap: "11,357 ignored vertices",
      },
      {
        label: "DOMINO20 clean v2",
        vertices: "20 / 20 assets",
        labeled: "220,692 positive / 45,972 negative",
        overlap: "116,112 ignored vertices",
      },
      {
        label: "All clean v2",
        vertices: "32 / 32 assets",
        labeled: "247,536 positive / 73,551 negative",
        overlap: "127,469 ignored vertices",
      },
    ],
    method: [
      "Build an asset manifest over DextoolBench12 and DOMINO20 objects.",
      "Render multi-view RGB images for each mesh and run the positive grasp prompt.",
      "Select the highest-quality SAM3 candidate per view instead of unioning all masks.",
      "Project 2D masks back to mesh vertices with visible-face voting.",
      "Apply connected-component and hole-fill cleanup to produce positive_grasp_only.npz.",
      "Reject extremely thin candidates for screwdriver and knife-like assets so shafts or blades are not mislabeled as handles.",
      "Build grasp_affordance_clean_v2.npz with 1 positive, 0 conservative negative, and -1 ignore.",
    ],
    visuals: [
      {
        title: "DextoolBench12 clean v2",
        src: "assets/images/affordance_dextoolbench12_clean_v2_overview.png",
        caption:
          "Clean binary v2 visualization for the DextoolBench objects. Blue/positive areas are preferred grasp regions; red/negative areas are discouraged.",
      },
      {
        title: "DOMINO20 clean v2",
        src: "assets/images/affordance_domino20_clean_v2_overview.png",
        caption:
          "Clean binary v2 labels for the 20 DOMINO objects used in the object-diversity branch.",
      },
      {
        title: "All clean v2 labels",
        src: "assets/images/affordance_all_clean_v2_overview.png",
        caption:
          "Combined overview of all 32 labeled assets for quick quality inspection.",
      },
    ],
    labelUse: [
      "Recommended training file: assets/affordance_labels/**/grasp_affordance_clean_v2.npz.",
      "Use grasp_label >= 0 as the valid mask; ignore -1 in supervised losses.",
      "For RL shaping, keep the scale small: positive contact bonus and conservative negative contact penalty.",
      "For the student, use clean-v2 labels as an affordance BCE auxiliary target over visible or projected object points.",
      "Do not use affordance labels as the final success definition.",
    ],
    quality: [
      {
        label: "More reliable",
        items: "mug, pill_bottle, can, plate, stapler, handle_eraser, blue_brush",
      },
      {
        label: "Use cautiously",
        items:
          "flat_eraser, sharpie_marker, milk_box, milk_tea, tea_box, apple, book, dumbbell",
      },
    ],
    files: [
      "assets/affordance_labels/**/positive_grasp_only.npz",
      "assets/affordance_labels/**/grasp_affordance_clean_v2.npz",
      "assets/affordance_labels/visualizations/*clean_v2_overview.png",
      "assets/affordance_labels/analysis/*clean_v2_analysis.csv",
      "scripts/run_positive_grasp_only_annotation.py",
      "scripts/analyze_affordance_annotations.py",
      "scripts/visualize_affordance_manifest.py",
    ],
  },
  experiments: [
    {
      version: "v36",
      title: "Franka+Inspire simple reward, no goal marker",
      status: "good",
      result: "strong visual baseline",
      body: "A useful Inspire reference where the policy reaches, closes, and lifts more reliably than later Revo2 attempts.",
    },
    {
      version: "v46",
      title: "Inspire privileged teacher",
      status: "good",
      result: "teacher reference",
      body: "Teacher-side point cloud branch with release-motion curriculum; still used as the behavior reference.",
    },
    {
      version: "v54-v58",
      title: "RGB-D temporal student branch",
      status: "watch",
      result: "not deployable yet",
      body: "Builds object point clouds from RGB-D render and mask over multiple frames. Training is slower and less stable than privileged teacher training.",
    },
    {
      version: "v87",
      title: "Revo2 tabletop physical-hand teacher",
      status: "active",
      result: "running on GPU7",
      body: "Uses the 11D physical Revo2 hand action space to test whether the coupled-hand bottleneck was blocking tabletop grasp learning.",
    },
    {
      version: "v88",
      title: "Revo2 Falling Baton physical catch",
      status: "active",
      result: "running on GPU5",
      body: "No-table falling-object catch setting with contact-like curriculum, predicted catch reward, catch-window reward, and post-contact velocity damping.",
    },
  ],
  videos: [
    {
      title: "Inspire simple-reward baseline",
      tag: "v36",
      tone: "good",
      result: "visual reference",
      src: "assets/videos/v36_inspire_can.mp4",
      poster: "assets/images/v36_inspire_can.jpg",
      caption: "Franka+Inspire behavior from the no-goal simple reward run.",
    },
    {
      title: "Inspire privileged teacher",
      tag: "v46",
      tone: "good",
      result: "teacher reference",
      src: "assets/videos/v46_inspire_teacher_can.mp4",
      poster: "assets/images/v46_inspire_teacher_can.jpg",
      caption: "Recent teacher-side run used as a reference when debugging Revo2.",
    },
    {
      title: "Revo2 stable-reach bootstrap",
      tag: "v72",
      tone: "watch",
      result: "insufficient lift",
      src: "assets/videos/v72_revo2_bottle.mp4",
      poster: "assets/images/v72_revo2_bottle.jpg",
      caption: "BrainCo Revo2 line after alignment fixes; still weak at closing and lifting.",
    },
    {
      title: "Revo2 lift-hold bridge current run",
      tag: "v73",
      tone: "watch",
      result: "active issue",
      src: "assets/videos/v73_revo2_bottle_current.mp4",
      poster: "assets/images/v73_revo2_bottle_current.jpg",
      caption: "Current Revo2 teacher run: contact rises, but successful lift-and-hold is still missing.",
    },
    {
      title: "Falling Baton physical catch",
      tag: "v88",
      tone: "active",
      result: "active run",
      src: "assets/videos/v88_falling_baton_screwdriver.mp4",
      poster: "assets/images/v88_falling_baton_screwdriver.jpg",
      caption: "Early V88 no-table falling-object catch video with the new front workspace and catch overlay.",
    },
    {
      title: "Revo2 aligned front preview",
      tag: "preview",
      tone: "reference",
      result: "embodiment",
      src: "assets/videos/revo2_aligned_front.mp4",
      poster: "assets/images/revo2_aligned_front.png",
      caption: "Front-view preview after aligning BrainCo Revo2 to the Inspire-style convention.",
    },
  ],
  diagnosis: [
    {
      title: "Revo2 is the current sim2real embodiment",
      body: "The page now treats Franka + BrainCo Revo2 as the primary branch while preserving Inspire and Sharpa as references.",
    },
    {
      title: "Falling Baton needs staged learning",
      body: "The V88 task uses a contact-like curriculum before requiring stable true grasp, because free-fall catching has a much shorter contact window than tabletop grasping.",
    },
    {
      title: "Student learning remains fragile",
      body: "RGB-D temporal point clouds are deployable in principle, but the lower point-cloud quality and camera cost make PPO slower and noisier.",
    },
    {
      title: "Affordance labels are weak priors",
      body: "Clean-v2 labels should guide contact and auxiliary prediction, but final success remains physical catch, grasp, lift, and stable hold.",
    },
  ],
};

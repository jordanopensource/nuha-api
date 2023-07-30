<!-- PROJECT LOGO -->
<div align="center">
<a href="https://github.com/jordanopensource/nuha-api">
  <img src=".github/images/logo.svg" alt="Logo" width="80" height="80">
</a>

### nuha-api

The community hub for JOSA.

[Explore the docs »](https://github.com/jordanopensource/nuha-api)

[Visit the Portal]()
.
[Report Bug](https://github.com/jordanopensource/nuha-api/issues)
.
[Request Feature](https://github.com/jordanopensource/nuha-api/issues)

[![Build Status](https://builds.josa.ngo/api/badges/jordanopensource/nuha-api/status.svg?ref=refs/heads/development)](https://builds.josa.ngo/jordanopensource/nuha-api)

</div>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#running">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This is a API client for the [Nuha Project](https://nuha.josa.ngo).

### Built With 🤖

___

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This project depends on a trained text classification model, which is hosted on [Hugging Face](https://huggingface.co/). You can either train your own model or use the one provided by JOSA. The model is defined in environment variables, which are passed to the application at runtime:
1. HUGGINGFACE_TOKEN: The Hugging Face API token.
2. MODEL_PATH: The model path on Hugging Face.
3. MODEL_VERSION: The model version on Hugging Face.

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/jordanopensource/nuha-api.git
   ```

2. Create a virtual environment

   ```sh
   python3 -m venv venv
   ```

3. Activate the virtual environment

   ```sh
    source venv/bin/activate
    ```

4. Install the dependencies

   ```sh
   pip install -r requirements.txt
   ```



### Running

#### Development

To run the project locally for development purposes:

1. Activate the virtual environment

   ```sh
    source venv/bin/activate
    ```

2. Run the project
   ```sh
    HUGGINGFACE_TOKEN="" MODEL_PATH="" MODEL_VERSION="" uvicorn app.main:app --reload
    ```

#### Production

To build and run the project locally for production purposes:

1. Build the Docker image

   ```sh
   docker build -t nuha-api .
   ```

2. Run the Docker container

   ```sh
    docker run -d -p 8000:8000 -e HUGGINGFACE_TOKEN="" -e MODEL_PATH="" -e MODEL_VERSION="" nuha-api
    ```



___

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/jordanopensource/nuha-api/issues) for a list of proposed features (and known issues).

___

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/amazing-feature`)
3. Commit your Changes (`git commit -m 'Add some amazing-feature'`)
4. Push to the Branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

___

<!-- LICENSE -->
## License

Distributed under the Apache License 2.0. See [LICENSE](LICENSE) for more information.

___

<!-- CONTACT -->
## Contact

Jordan Open Source Association - [@jo_osa](https://twitter.com/@jo_osa)

Project Link: [https://github.com/jordanopensource/nuha-api](https://github.com/jordanopensource/nuha-api)

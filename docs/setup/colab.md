# Colab T4 setup

The data scientist runs GPU notebooks on a Colab T4 through the official
[Google Colab CLI](https://github.com/googlecolab/google-colab-cli). How a run works is
in [the data scientist doc](../agents/data-scientist.md#a-colab-t4-run).

1. The dev container installs the Colab CLI and the
   [Google Cloud CLI](https://cloud.google.com/sdk/docs/install). Outside it, run
   `make colab-setup` and install the Google Cloud CLI yourself.
2. Sign in once with the Google account you use with Colab: run `make colab-login`, open
   the link it prints, and paste the code back. It then checks access with
   `colab --auth adc usage` (or run `make colab-check` any time).
3. In a chat with Jack or the data scientist, open **Settings → Notebook compute**, select
   **Colab T4** and save. New chats start with **Local CPU**.

The sign-in is stored in `/workspaces/.gcloud` (`CLOUDSDK_CONFIG`), outside the repo, so
it survives container rebuilds in Codespaces and is never committed. Deleting the
codespace removes it; a local dev container needs a new sign-in after a rebuild.

Kaggle data is downloaded with `kagglehub` from public dataset URLs; no Kaggle key is
stored anywhere. Competition files you are allowed to use with Colab can be uploaded to
the chat instead.

from gpt import DataLoader, Transformer, Config
import torch

if __name__ == "__main__":
    cfg = Config()
    model = Transformer()
    state_dict = torch.load("./Model/final.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    text = """
Merchant of Syracuse, plead no more;
I am not partial to infringe our laws:
The enmity and discord which of late
Sprung from the rancorous outrage of your duke
To merchants, our well-dealing countrymen,
Who wanting guilders to redeem their lives
Have seal'd his rigorous statutes with their bloods,
Excludes all pity from our threatening looks.
For, since the mortal and intestine jars
'Twixt thy seditious countrymen and us,
It hath in solemn synods been decreed
Both by the Syracusians and ourselves,
To admit no traffic to our adverse towns Nay, more,
If any born at Ephesus be seen
At any Syracusian marts and fairs;
Again: if any Syracusian born
    """
    print(
        "".join(
            model.generate(text, DataLoader("input.txt", Config()), max_new_token=500)[
                0
            ]
        )
    )

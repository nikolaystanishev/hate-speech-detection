import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from core.model import PretrainedModel
import clip
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans
import numpy as np

def makeClipDataset(new_data_file_path, old_data_file_path):
	if not os.path.exists(old_data_file_path):
		print('file not found:', old_data_file_path)
		return
	if os.path.exists(new_data_file_path):
		print('file already exists, remove it to proceed:', new_data_file_path)
		return
	data_dir = os.path.dirname(old_data_file_path)
	model, transform = PretrainedModel.load_clip_model()
	tokenizer = PretrainedModel.load_clip_tokenizer()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device=device)
	data = [json.loads(jline) for jline in open(old_data_file_path, 'r').readlines()]

	with open(new_data_file_path, 'w') as f:
		for line in tqdm(data, total=len(data)):
			text = line['text']
			text = tokenizer(text, context_length=77, truncate=True).to(device=device)
			text = model.encode_text(text).to(dtype=torch.float32)
			img = Image.open(os.path.join(data_dir, line['img']))
			img = transform(img).unsqueeze(0).to(device=device)
			img = model.encode_image(img).to(dtype=torch.float32)
			f.write(json.dumps({'img_embedding': img.cpu().detach().numpy().tolist(), 'text_embedding': text.cpu().detach().numpy().tolist(), 'label': line['label']}) + '\n')

makeClipDataset('data/train_clip.jsonl', 'data/train.jsonl')
makeClipDataset('data/eval_clip.jsonl', 'data/dev.jsonl')
makeClipDataset('data/test_clip.jsonl', 'data/test.jsonl')


def saveClipDatasetEmbeddings(data_file_path, classes):
	if os.path.exists(data_file_path):
		print('file already exists, remove it to proceed:', data_file_path)
		#return

	with open(data_file_path, 'w') as f:
		for c in tqdm(classes, total=len(classes)):
			text = c['embedding']
			f.write(json.dumps({'embedding': text.cpu().detach().numpy().tolist()}) + '\n')

classes = ['Hate Speech', 'Insult', 'Sarcastic', 'Informative', 'Blasphemy', 'Swearing', 'Threat']
#classes += ['Ass', 'Asshole', 'Homosexual', 'Homophobic', 'Racist', 'Gay', 'Lgbt', 'Jew', 'Jewish', 'Anti-semitic', 'Chink', 'Muslims', 'Muslim', 'Isis', 'Islamophobe', 'homophobe ', 'Bombing', 'Sexyhot', 'Bastard', 'Bitch', 'Fucker', 'Cunt', 'Damn', 'Fuck', 'Goddamn', 'Shit', 'Motherfucker', 'Nigga', 'Nigger', 'Prick', 'Shit', 'shit ass', 'Shitass', 'son of a bitch', 'Whore', 'Thot', 'Slut', 'Faggot', 'Dick', 'Pussy', 'Penis', 'Vagina', 'Negro', 'Coon', 'Bitched', 'Sexist', 'Freaking', 'Cock', 'Sucker', 'Lick', 'Licker', 'Rape', 'Molest', 'Anal', 'Buttrape', 'Coont', 'Cancer', 'Sex', 'Retard', 'Fuckface', 'Dumbass', '5h1t', '5hit', 'A_s_s', 'a2m', 'a55', 'adult', 'amateur', 'anal', 'anal impaler', 'anal leakage', 'anilingus', 'anus', 'ar5e', 'arrse', 'arse', 'arsehole', 'ass', 'ass fuck', 'asses', 'assfucker', 'ass-fucker', 'assfukka', 'asshole', 'asshole', 'assholes', 'assmucus', 'assmunch', 'asswhole', 'autoerotic', 'b!tch', 'b00bs', 'b17ch', 'b1tch', 'ballbag', 'ballsack', 'bang (one\'s) box', 'bangbros', 'bareback', 'bastard', 'beastial', 'beastiality', 'beef curtain', 'bellend', 'bestial', 'bestiality', 'bi+ch', 'biatch', 'bimbos', 'birdlock', 'bitch', 'bitch tit', 'bitcher', 'bitchers', 'bitches', 'bitchin', 'bitching', 'bloody', 'blow job', 'blow me', 'blow mud', 'blowjob', 'blowjobs', 'blue waffle', 'blumpkin', 'boiolas', 'bollock', 'bollok', 'boner', 'boob', 'boobs', 'booobs', 'boooobs', 'booooobs', 'booooooobs', 'breasts', 'buceta', 'bugger', 'bum', 'bunny fucker', 'bust a load', 'busty', 'butt', 'butt fuck', 'butthole', 'buttmuch', 'buttplug', 'c0ck', 'c0cksucker', 'carpet muncher', 'carpetmuncher', 'cawk', 'chink', 'choade', 'chota bags', 'cipa', 'cl1t', 'clit', 'clit licker', 'clitoris', 'clits', 'clitty litter', 'clusterfuck', 'cnut', 'cock', 'cock pocket', 'cock snot', 'cockface', 'cockhead', 'cockmunch', 'cockmuncher', 'cocks', 'cocksuck ', 'cocksucked ', 'cocksucker', 'cock-sucker', 'cocksucking', 'cocksucks ', 'cocksuka', 'cocksukka', 'cok', 'cokmuncher', 'coksucka', 'coon', 'cop some wood', 'cornhole', 'corp whore', 'cox', 'cum', 'cum chugger', 'cum dumpster', 'cum freak', 'cum guzzler', 'cumdump', 'cummer', 'cumming', 'cums', 'cumshot', 'cunilingus', 'cunillingus', 'cunnilingus', 'cunt', 'cunt hair', 'cuntbag', 'cuntlick ', 'cuntlicker ', 'cuntlicking ', 'cunts', 'cuntsicle', 'cunt-struck', 'cut rope', 'cyalis', 'cyberfuc', 'cyberfuck ', 'cyberfucked ', 'cyberfucker', 'cyberfuckers', 'cyberfucking ', 'd1ck', 'damn', 'dick', 'dick hole', 'dick shy', 'dickhead', 'dildo', 'dildos', 'dink', 'dinks', 'dirsa', 'dirty Sanchez', 'dlck', 'dog-fucker', 'doggie style', 'doggiestyle', 'doggin', 'dogging', 'donkeyribber', 'doosh', 'duche', 'dyke', 'eat a dick', 'eat hair pie', 'ejaculate', 'ejaculated', 'ejaculates ', 'ejaculating ', 'ejaculatings', 'ejaculation', 'ejakulate', 'erotic', 'f u c k', 'f u c k e r', 'f_u_c_k', 'f4nny', 'facial', 'fag', 'fagging', 'faggitt', 'faggot', 'faggs', 'fagot', 'fagots', 'fags', 'fanny', 'fannyflaps', 'fannyfucker', 'fanyy', 'fatass', 'fcuk', 'fcuker', 'fcuking', 'feck', 'fecker', 'felching', 'fellate', 'fellatio', 'fingerfuck ', 'fingerfucked ', 'fingerfucker ', 'fingerfuckers', 'fingerfucking ', 'fingerfucks ', 'fist fuck', 'fistfuck', 'fistfucked ', 'fistfucker ', 'fistfuckers ', 'fistfucking ', 'fistfuckings ', 'fistfucks ', 'flange', 'flog the log', 'fook', 'fooker', 'fuck hole', 'fuck puppet', 'fuck trophy', 'fuck yo mama', 'fuck', 'fucka', 'fuck-ass', 'fuck-bitch', 'fucked', 'fucker', 'fuckers', 'fuckhead', 'fuckheads', 'fuckin', 'fucking', 'fuckings', 'fuckingshitmotherfucker', 'fuckme ', 'fuckmeat', 'fucks', 'fucktoy', 'fuckwhit', 'fuckwit', 'fudge packer', 'fudgepacker', 'fuk', 'fuker', 'fukker', 'fukkin', 'fuks', 'fukwhit', 'fukwit', 'fux', 'fux0r', 'gangbang', 'gangbang', 'gang-bang', 'gangbanged ', 'gangbangs ', 'gassy ass', 'gaylord', 'gaysex', 'goatse', 'god', 'god damn', 'god-dam', 'goddamn', 'goddamned', 'god-damned', 'ham flap', 'hardcoresex ', 'hell', 'heshe', 'hoar', 'hoare', 'hoer', 'homo', 'homoerotic', 'hore', 'horniest', 'horny', 'hotsex', 'how to kill', 'how to murdep', 'jackoff', 'jack-off ', 'jap', 'jerk', 'jerk-off ', 'jism', 'jiz ', 'jizm ', 'jizz', 'kawk', 'kinky Jesus', 'knob', 'knob end', 'knobead', 'knobed', 'knobend', 'knobend', 'knobhead', 'knobjocky', 'knobjokey', 'kock', 'kondum', 'kondums', 'kum', 'kummer', 'kumming', 'kums', 'kunilingus', 'kwif', 'l3i+ch', 'l3itch', 'labia', 'LEN', 'lmao', 'lmfao', 'lmfao', 'lust', 'lusting', 'm0f0', 'm0fo', 'm45terbate', 'ma5terb8', 'ma5terbate', 'mafugly', 'masochist', 'masterb8', 'masterbat*', 'masterbat3', 'masterbate', 'master-bate', 'masterbation', 'masterbations', 'masturbate', 'mof0', 'mofo', 'mo-fo', 'mothafuck', 'mothafucka', 'mothafuckas', 'mothafuckaz', 'mothafucked ', 'mothafucker', 'mothafuckers', 'mothafuckin', 'mothafucking ', 'mothafuckings', 'mothafucks', 'mother fucker', 'mother fucker', 'motherfuck', 'motherfucked', 'motherfucker', 'motherfuckers', 'motherfuckin', 'motherfucking', 'motherfuckings', 'motherfuckka', 'motherfucks', 'muff', 'muff puff', 'mutha', 'muthafecker', 'muthafuckker', 'muther', 'mutherfucker', 'n1gga', 'n1gger', 'nazi', 'need the dick', 'nigg3r', 'nigg4h', 'nigga', 'niggah', 'niggas', 'niggaz', 'nigger', 'niggers ', 'nob', 'nob jokey', 'nobhead', 'nobjocky', 'nobjokey', 'numbnuts', 'nut butter', 'nutsack', 'omg', 'orgasim ', 'orgasims ', 'orgasm', 'orgasms ', 'p0rn', 'pawn', 'pecker', 'penis', 'penisfucker', 'phonesex', 'phuck', 'phuk', 'phuked', 'phuking', 'phukked', 'phukking', 'phuks', 'phuq', 'pigfucker', 'pimpis', 'piss', 'pissed', 'pisser', 'pissers', 'pisses ', 'pissflaps', 'pissin ', 'pissing', 'pissoff ', 'poop', 'porn', 'porno', 'pornography', 'pornos', 'prick', 'pricks ', 'pron', 'pube', 'pusse', 'pussi', 'pussies', 'pussy', 'pussy fart', 'pussy palace', 'pussys ', 'queaf', 'queer', 'rectum', 'retard', 'rimjaw', 'rimming', 's hit', 's.o.b.', 's_h_i_t', 'sadism', 'sadist', 'sandbar', 'sausage queen', 'schlong', 'screwing', 'scroat', 'scrote', 'scrotum', 'semen', 'sex', 'sh!+', 'sh!t', 'sh1t', 'shag', 'shagger', 'shaggin', 'shagging', 'shemale', 'shi+', 'shit', 'shit fucker', 'shitdick', 'shite', 'shited', 'shitey', 'shitfuck', 'shitfull', 'shithead', 'shiting', 'shitings', 'shits', 'shitted', 'shitter', 'shitters ', 'shitting', 'shittings', 'shitty ', 'skank', 'slope', 'slut', 'slut bucket', 'sluts', 'smegma', 'smut', 'snatch', 'son-of-a-bitch', 'spac', 'spunk', 't1tt1e5', 't1tties', 'teets', 'teez', 'testical', 'testicle', 'tit', 'tit wank', 'titfuck', 'tits', 'titt', 'tittie5', 'tittiefucker', 'titties', 'tittyfuck', 'tittywank', 'titwank', 'tosser', 'turd', 'tw4t', 'twat', 'twathead', 'twatty', 'twunt', 'twunter', 'v14gra', 'v1gra', 'vagina', 'viagra', 'vulva', 'w00se', 'wang', 'wank', 'wanker', 'wanky', 'whoar', 'whore', 'willies', 'willy', 'wtf', 'xrated', 'xxx', 'sucker', 'dumbass', 'Kys', 'Kill', 'Die', 'Cliff', 'Bridge', 'Shooting', 'Shoot', 'Bomb', 'Terrorist', 'Terrorism', 'Bombed', 'Trump', 'Maga', 'Conservative', 'Make america great again', 'Far right', 'Necrophilia', 'Mongoloid', 'Furfag', 'Cp', 'Pedo', 'Pedophile', 'Pedophilia', 'Child predator', 'Predatory', 'Depression', 'Cut myself', 'I want to die', 'Fuck life', 'Redtube', 'Loli', 'Lolicon', 'Cub']
classes_embeddings = []

model = PretrainedModel.load_clip_model()[0]
for c in tqdm(classes, total=len(classes)):
	c_embedded = model.encode_text(clip.tokenize(c)).to(dtype=torch.float32)
	classes_embeddings.append({'text': c, 'embedding': c_embedded})
saveClipDatasetEmbeddings('data/classes_clip.jsonl', classes_embeddings)



# X = np.array([el.cpu().detach().numpy().tolist() for el in classes_embeddings])
# kmeans = KMeans(n_clusters=100, random_state=0, n_init="auto").fit(X)
# kmeans.labels_
# kmeans.predict([[0, 0], [12, 3]])
# kmeans.cluster_centers_

import pandas as pd
if __name__ == "__main__":
	uname=['user_id','gender','age','occupation','zip']
	users=pd.read_table(r'users.dat',sep='::',header=None,names=uname,engine = 'python')
	rnames=['user_id','movie_id','rating','timestamp']
	ratings=pd.read_table(r'ratings.dat',sep='::',header=None,names=rnames,engine = 'python')
	mname=['movie_id','title','genres']
	movies=pd.read_table(r'movies.dat',sep='::',header=None,names=mname,engine = 'python')
	features = [ "movie_id","user_id","gender","age","occupation","zip"]
	target = ['rating']
	data=pd.merge(pd.merge(ratings,movies,how='left',on='movie_id'),users,how='left',on='user_id')
	data = data.sample(frac=0.01,random_state=1024).to_pickle("movielens_sample.pkl")
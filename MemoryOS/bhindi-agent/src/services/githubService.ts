/**
 * GitHub service for interacting with the GitHub API
 */
export class GitHubService {
  private baseUrl = 'https://api.github.com';

  /**
   * List repositories for the authenticated user
   * @param token GitHub personal access token
   * @param options Query parameters for the request
   * @returns List of repositories
   */
  async listUserRepositories(token: string, options: {
    per_page?: number;
    sort?: 'created' | 'updated' | 'pushed' | 'full_name';
    direction?: 'asc' | 'desc';
    type?: 'all' | 'owner' | 'public' | 'private' | 'member';
  } = {}) {
    const { per_page = 10, sort = 'updated', direction = 'desc', type = 'owner' } = options;

    try {
      const response = await fetch(`${this.baseUrl}/user/repos?per_page=${per_page}&sort=${sort}&direction=${direction}&type=${type}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/vnd.github.v3+json',
          'Authorization': `token ${token}`,
          'User-Agent': 'Bhindi-Agent'
        }
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `GitHub API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      
      // Extract relevant information from each repository
      const repositories = data.map((repo: any) => ({
        id: repo.id,
        name: repo.name,
        full_name: repo.full_name,
        description: repo.description,
        html_url: repo.html_url,
        language: repo.language,
        stargazers_count: repo.stargazers_count,
        forks_count: repo.forks_count,
        created_at: repo.created_at,
        updated_at: repo.updated_at,
        visibility: repo.visibility,
        default_branch: repo.default_branch
      }));

      return {
        count: repositories.length,
        repositories,
        query_params: { per_page, sort, direction, type }
      };
    } catch (error) {
      throw error instanceof Error 
        ? error 
        : new Error('Unknown error occurred while accessing GitHub API');
    }
  }
}
